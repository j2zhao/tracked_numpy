from compression import compression
import numpy as np
import numpy.core.tracked_float as tf
import inspect
import json
import copy
import sys 
import constants
import os

def _check_equality(val1, val2):
    if not isinstance(val1, list) and not isinstance(val2, list):
        if val1 != val2:
            return None
        else:
            return val1
    elif isinstance(val1, list) and not isinstance(val2, list):
        if val2 in val1:
            return val2
        else:
            return None
    elif not isinstance(val1, list) and isinstance(val2, list):
        if val1 in val2:
            return val1
        else:
            return None
    else:
        vals = []
        for v1 in val1:
            if v1 in val2:
                vals.append(v1)
        if len(vals) == 0:
            return None
        elif len(vals) == 1:
            return vals[0]
        else:
            return vals

def _check_equality2(val1, val2):
    a = _check_equality(val1[0], val2[0])
    b = _check_equality(val1[1], val2[1])
    if a == None or b == None:
        return None
    else:
        return (a, b)

def _check_equality_prov(pr1, pr2):
    if len(pr1) != len(pr2):
        return None
    
    pr = []
    for i in range(len(pr1)):
        p = _check_equality(pr1[i], pr2[i])
        if p == None:
            return None
        else:
            pr.append(p)
    return pr

def check_eq_prov(prov1, prov2):
    if prov1 == constants.UNKNOWN or prov2 == constants.UNKNOWN:
        return constants.UNKNOWN
    elif prov1 == prov2:
        return prov1

    new_prov = {}
    for i in prov1:
        if i not in prov2:
            return constants.UNKNOWN
        if len(prov1[i]) != len(prov2[i]):
            return constants.UNKNOWN
        new_prov[i] = []
        for j in range(len(prov1[i])):
            a1, b1, p1 = prov1[i][j]
            a2, b2, p2 = prov2[i][j]
            a = _check_equality2(a1, a2)
            if a == None:
                return constants.UNKNOWN
            b = _check_equality2(b1, b2)
            if b == None:
                return constants.UNKNOWN
            p = []
            for d in range(len(p1)):
                p.append({})
                for pname in p1[d]:
                    if pname in p2[d]:
                        if p1[d][pname] == p2[d][pname]:
                            p[d][pname] = p1[d][pname]
                        else:
                            temp = _check_equality_prov(p1[d][pname], p2[d][pname])
                            if temp != None:
                                p[d][pname] = temp
                if len(p[d]) == 0:
                    return constants.UNKNOWN
            
            new_prov[i].append((a, b, p))
    
    return new_prov


# deal with list of arrays -> TODO later
# arguments can't change type
class FunctionProvenance():
    """
    provenance tuple:
    [args_list, provenance, number of passes, array_shapes]
    
    """
    def __init__(self, log, n = 2, m = 2, ratio = 0.1, saved_prov = None, new_cur = True):
        # if saved_function:
        #     with open(saved_function, 'r') as f:
        #         self.prov = json.load(f)
        # else:
        #     self.prov = {}

        if saved_prov:
            self.prov = saved_prov
        else:
            self.prov = {}
        if not new_cur:
            for func in self.prov:
                self.prov[func]['cur_provs'] = 0
        if not os.path.exists(log):
            with open(log, 'w'): pass
        self.log = log
        self.ratio = ratio
        self.n = n
        self.m = m

    def _run_func(self, func, args, kwargs):
        
        # initialize inputs
        oargs = []
        i = 0
        for arr in args:
            if isinstance(arr, np.ndarray):
                print(arr.shape)
                array = arr.astype(tf.tracked_float)
                tf.initialize(array, i)
                oargs.append(array)
                i += 1
            else:
                oargs.append(arr)
        okwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                okwargs[k] = tf.initialize(v.astype(tf.tracked_float), i)
                i += 1
            else:
                okwargs[k] = v 
        
        # run function
        #output = func(oargs, axis = 0) 
        output = func(*oargs, **okwargs)
        return output


    def _new_function(self, func):
        #args = inspect.getfullargspec(func)
        nfunc = func.__name__
        self.prov[nfunc] = {}
        #self.prov[nfunc]['args'] = args
        self.prov[nfunc]['provs'] = []
        self.prov[nfunc]['val_args'] = []
        self.prov[nfunc]['arb_args'] = []

        # size array related
        self.prov[nfunc]['arr_args'] = []
        # related to current program
        # provenance tuple:
        #{ARRAY_ARGS: [(args_list, provenance, number of passes)]}
        self.prov[nfunc]['cur_provs'] = {}
    
    def _get_args(self, nfunc, args, kwargs):
        # update with shape
        fargs = {}        
        for i, arg in enumerate(args):
            fargs[i] = arg
        
        for k in kwargs:
            fargs[k] = kwargs[k]
        
        arg_arrs = self.prov[nfunc]['arr_args']
        arrs_shape = []
        for key, val in fargs.items():
            if isinstance(val, np.ndarray):
                if key not in arg_arrs:
                    arg_arrs.append(key)
                arrs_shape.append((key, val.shape))
                fargs[key] = constants.ARRAY
        return fargs, tuple(arrs_shape)

    def prov_function(self, func, args, kwargs):
        ''''
        Returns: output of function, compressed provenance (if existing), version in database
        0: not existing
        1: existing but less than min passes
        2: existing but stated as unknown
        3: existing and returned

        0 -> general
        1 -> specific
        '''
        type = -1
        #initialize new function dictionary
        nfunc = func.__name__
        if nfunc not in self.prov:
            self._new_function(func)

        # get argument and set with arbitrary values
        arg_dic, arr_tup = self._get_args(nfunc, args, kwargs)
        
        provs = self.prov[nfunc]['provs']
        
        aarg_dic = copy.deepcopy(arg_dic)
        arbs = self.prov[nfunc]['arb_args']
        for a in arbs:
            if a in aarg_dic:
                aarg_dic[a] = constants.ARB
        # try to find match in general provenance (linear right now -> probably good enough)
        provenance = None
        num_pass = 0
        for prov in provs:
            if prov[0] == aarg_dic:
                provenance = prov[1]
                num_pass = prov[2]
        
        if provenance != None and num_pass > self.n and provenance != constants.UNKNOWN:
            type = 0
        # try to find match in specific provenance
        else:
            provenance = None
            cur_provs = self.prov[nfunc]['cur_provs']
            if arr_tup in cur_provs:
                for prov in cur_provs[arr_tup]:
                    if prov[0] == arg_dic:
                        provenance = prov[1]
                        num_pass = prov[2]
            if provenance != None:
                type = 1
        #raise ValueError()

        # if there isn't a match, we should run the full program
        if provenance == None:
            output = self._run_func(func, args, kwargs)
            return arg_dic, arr_tup, output, None, 0, type

        # if there is a match but the number of times we've seen a program is < n, we should run the number program
        elif num_pass < self.n:
            output = self._run_func(func, args, kwargs)
            return arg_dic, arr_tup, output, None, 1, type

        # if there is a match but we couldn't find a pattern
        elif provenance == constants.UNKNOWN:
            output = self._run_func(func, args, kwargs)
            return arg_dic, arr_tup, output, None, 2, type
        
        # return just the function if there is a match
        else:
            output = func(*args, **kwargs)
            return arg_dic, arr_tup, output, provenance, 3, type

    # test if getsize of works
    def compress_function(self, prov):
        compress_prov = compression(prov)
        return compress_prov


    def _gen_tup(self, tup, dict_arr):
        a, b = tup
        if a in dict_arr and b in dict_arr:
            return (dict_arr[a], dict_arr[b])
        elif a in dict_arr:
            return (dict_arr[a], b)
        elif b in dict_arr:
            return (a, dict_arr[b])
        else:
            return (a, b)
    
    def arb_size(self, provenance, arr_tup):
        if provenance == constants.UNKNOWN:
            return provenance
        dict_arr = {}
        for tup in arr_tup:
            key, shape = tup
            a = shape[0] - 1
            if a not in dict_arr:
                dict_arr[shape[0]-1] = [a]
            dict_arr[a].append(str(key) + '0')

            b = shape[1] - 1
            if b not in dict_arr:
                dict_arr[b] = [b]
            dict_arr[b].append(str(key) + '1')

        for id in provenance:
            for i, val in enumerate(provenance[id]):
                w, h, prov = val
                w = self._gen_tup(w, dict_arr)
                h = self._gen_tup(h, dict_arr)
                for d in range(len(prov)):
                    for key in prov[d]:
                        for j, val2 in enumerate(prov[d][key]):
                            a = self._gen_tup(val2, dict_arr)
                            prov[d][key][j] = a
                provenance[id][i] = (w, h, prov)
        return provenance


    def add_prov(self, provenance, nfunc, args, arr_args, add_arb = True):
        provs = self.prov[nfunc]['provs']
        arb_vals = self.prov[nfunc]['arb_args']
        cur_provs = self.prov[nfunc]['cur_provs']
        found = False
        
        arb_args = {}
        for arg in args:
            if arg in arb_vals:
                arb_args[arg] = constants.ARB
            else:
                arb_args[arg] = args[arg]
        rel_prov = self.arb_size(copy.deepcopy(provenance), arr_args)
        for prov in provs:
            if prov[0] == arb_args:
                equal = check_eq_prov(prov[1], rel_prov)
                if equal == None:
                    prov[1] = constants.UNKNOWN
                else:
                    prov[1] = equal
                if arr_args not in prov[3]:
                    prov[2] += 1
                    prov[3].append(arr_args)
                found = True
                break
        if not found:
            full_provenance = [arb_args, rel_prov, 1, [arr_args]]
            self.prov[nfunc]['provs'].append(full_provenance)

        found_2 = False
        if arr_args in cur_provs:
            for prov in cur_provs[arr_args]:
                if prov[0] == args:
                    if prov[1] != provenance:
                        prov[1] = constants.UNKNOWN
                    prov[2] += 1
                    found_2 = True
                    break
            if not found_2:
                full_provenance = [arb_args, provenance, 1]
                self.prov[nfunc]['cur_provs'][arr_args].append(full_provenance)
        else:
            full_provenance = [args, provenance, 1]
            self.prov[nfunc]['cur_provs'][arr_args] = [full_provenance]
        
        if add_arb:
            self.update_arbitrary(nfunc, arb_args, rel_prov)
        return arb_args, found
    
    def add_log(self, line, arrays, nfunc, args, arr_tup, prov):
        ### NEED TO DEAL WITH ARRAYS MAINLY
        # if isinstance(prov, np.array):
        #     array_name = ''
        #     np.save(array_name, prov) #need to generate new array with provenance?
        #     with open(self.log) as f:
        #         tup = (line, arrays, nfunc, args, array_name, 0)
        #         f.write(str(tup)) # does this work?  i might need to convert customly?
        with open(self.log, 'a') as f:
            tup = (line, arrays, nfunc, args, arr_tup, prov)
            f.write(str(tup)) # does this work?  i might need to convert customly??
    
    def save(self, file_name):
        with open(file_name, 'a') as f:
            json.dump(self.prov, f)

    def _update_arbitrary(self, nfunc, arb_list):
        '''this function checks it with all the other ones
        this is O(prov^2) because of the fact that arguments can be mutable -> preserve functionality over speed '''
        provs = self.prov[nfunc]['provs']        
        #find a list of qualified values
        failed_args = []
        
        # check every argument
        for arg in arb_list:
            failed = False
            #for every provenance convert to arbitrary
            new_provs = []
            for prov in provs:
                if arg in prov[0]:
                    unique = True
                    prov_args = copy.deepcopy(prov[0])
                    prov_args[arg] = constants.ARB
                    # this is the expensive check?
                    for prov2 in new_provs:
                        if prov2[0] == prov_args:
                            unique = False
                            prov_val = check_eq_prov(prov2[1], prov[1])
                            prov2 = prov_val
                            if prov_val == constants.UNKNOWN:
                                failed_args.append(arg)
                                if arg not in self.prov[nfunc]['val_args']:
                                    self.prov[nfunc]['val_args'].append(arg)
                                failed = True
                                break
                    if failed:
                        break
                    if unique:
                        new_provs.append((prov_args, prov[1], prov[2], prov[3]))
        # find new list of arbitrary args
        success = []
        for arg in arb_list:
            if arg not in failed_args:
                success.append(arg)
        new_provs = []
        for prov in provs:
            for arg in prov[0]:
                if arg in success:
                    prov[0][arg] = constants.ARB
            
            dup = False
            for prov2 in new_provs:
                if prov[0] == prov2[0]:
                    prov2[1] = check_eq_prov(prov2[1], prov[1])
                    dup = True
                    prov[3].extend(x for x in prov2[3] if x not in prov[3])
                    prov[2] = len(prov[3])
            if not dup:
                new_provs.append(prov)
        
        self.prov[nfunc]['provs'] = new_provs
        self.prov[nfunc]['arb_args'] += success
        return new_provs, success

    def update_arbitrary(self, nfunc, args, provenance):
        ''' '''
        provs = self.prov[nfunc]['provs']
        val_args = self.prov[nfunc]['val_args']
        arb_list = self.prov[nfunc]['arb_args']
        arr_list = self.prov[nfunc]['arr_args']
        arg_count = {}
        
        for prov in provs:
            p = check_eq_prov(prov[1], provenance)
            if p != constants.UNKNOWN:
                for k in args:
                    if k in val_args or k in arb_list or k in arr_list:
                        continue
                    if k in prov[0] and args[k] != prov[0][k]:
                        if k in arg_count:
                            arg_count[k] += 1
                        else:
                            arg_count[k] = 1
        arb_list = []
        for arg in arg_count:
            if arg_count[arg] >= self.m:
                arb_list.append(arg)
        arb_list = self._update_arbitrary(nfunc, arb_list)




    


