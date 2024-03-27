import torch
import multiprocessing as mp
import numpy as np

import lmdb
import lz4.frame
import os
import logging
import numpy as np
import time
import math
from typing import List

from dblogger import DB_logger

import secrets

from typing import Dict, List

import pickle

DEBUG_MODE = True

# import sys
# sys.path.append('./build/lib.linux-x86_64-cpython-310')
import cpp_dearth_dataloader


def clear_shm():
    # remove everything in /dev/shm that starts with the prefix
    prefix = cpp_dearth_dataloader.get_shm_prefix()
    print(f"clear_shm: prefix = {prefix}")
    for filename in os.listdir("/dev/shm"):
        if filename.startswith(prefix):
            os.remove(os.path.join("/dev/shm", filename))
            
def clear_shm_start_with(prefix):
    # remove everything in /dev/shm that starts with the prefix
    prefix = cpp_dearth_dataloader.get_shm_prefix() + prefix
    print(f"clear_shm: prefix = {prefix}")
    for filename in os.listdir("/dev/shm"):
        if filename.startswith(prefix):
            os.remove(os.path.join("/dev/shm", filename))
            

def num_to_base36(num: int) -> str:
    base64_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ret = ""
    if num == 0:
        return "0"
    while num > 0:
        ret = base64_chars[num % 36] + ret
        num = num // 36
    return ret


class Dataloader:
    instance_cnt = 0
    
    def __init__(self, data_rootpath: str, tokenizer, config: object, seqlen: int, worker_num=None, log_dir=None, log_name=None, need_clear_shm=True):
        # check data_rootpath exist and is a directory
        if not os.path.exists(data_rootpath):
            raise RuntimeError(f"data_rootpath {data_rootpath} does not exist")
        if not os.path.isdir(data_rootpath):
            raise RuntimeError(f"data_rootpath {data_rootpath} is not a directory")
        
        self.instance_id = num_to_base36(Dataloader.instance_cnt) + '-' + num_to_base36(secrets.randbelow(100000))
        if Dataloader.instance_cnt == 0 and need_clear_shm: # only clear all shm for the first instance
            clear_shm()
        Dataloader.instance_cnt += 1
        
        if worker_num is None:
            worker_num = max(int(mp.cpu_count()*0.6), mp.cpu_count()-12, 8)

        self.is_alive = True
        
        self.history_time = []
        self.lifetime_history_time = []
        
        self.current_batch_id = 0
        self.recent_seqlen = seqlen
        self.data_rootpath = data_rootpath
        
        self.db_logger = None
        if log_dir is not None and log_name is not None:
            self.db_logger = DB_logger(logger_name=log_name, log_dir=log_dir)
            logging.info(f"Dataloader will log to {log_dir}/{log_name}")
        
        # creates workers
        self.workers = []
        self.check_and_compute_dataset(config)
        for i in range(worker_num):
            self.workers.append(mp.Process(target=_worker_main, args=(data_rootpath, self.db_names, 
                                                                      tokenizer, i, self.instance_id,
                                                                      log_dir, log_name)))
        for worker in self.workers:
            worker.start()
        
        # create manager in cpp
        self.manager = cpp_dearth_dataloader.DearthDataloaderManagerCpp(self.db_names, self.db_weights, worker_num, self.instance_id)
        

    def get_batch(self, batch_size, seqlen=None) -> torch.Tensor:
        start_time = time.time()
        self.current_batch_id += 1
        
        if seqlen is None:
            seqlen = self.recent_seqlen
        
        fail_cnt = 0
        while True:
            ret_np: np.ndarray = self.manager.get_batch(batch_size, seqlen)
            if ret_np is not None:
                break
            else:
                fail_cnt += 1
                if fail_cnt % 20 == 1 and self.current_batch_id > 10:
                    logging.warn(f"get_batch: batch_id = {self.current_batch_id}, failed to get batch, retrying")
                time.sleep(0.01)
        
        #ret = torch.tensor(ret_np, dtype=torch.int64)
        ret = torch.from_numpy(ret_np).to(torch.int64)
        
        end_time = time.time()
        self.history_time.append(end_time - start_time)
        self.lifetime_history_time.append(end_time - start_time)
        #logging.debug(f"get_batch: {end_time - start_time}")
        if len(self.history_time) > 1000:
            #logging.debug(f"average time for get_batch: {sum(self.history_time)/len(self.history_time)}")
            self.history_time = []
        return ret
    
    def check_and_compute_dataset(self, config):
        """
        check if the dataset exists and is a directory,
        and compute the probability to sample from each dataset, save the weight to self.dataset_prob
        """
        db_names = list(config.keys())
        for name in db_names:
            if len(name) > 100:
                raise RuntimeError(f"dataset name {name} is too long, should be less than 100 characters due to the char[128] in struct in cpp size")
            dataset_path = os.path.join(self.data_rootpath, name)
            if not os.path.exists(dataset_path):
                logging.error(f"dataset {name} does not exist")
                config.pop(name)
                continue
            if not os.path.isdir(dataset_path):
                logging.error(f"dataset {name} is not a directory")
                config.pop(name)
                continue
            assert config[name] > 0, f"dataset {name} has weight <= 0"
        
        self.db_names = []
        self.db_weights = []
        for name in config:
            self.db_names.append(name)
            self.db_weights.append(config[name])
            
        if len(self.db_names) == 0:
            raise RuntimeError("no dataset to load")
    

    def destory(self):
        if not self.is_alive:
            return
        
        self.manager.destory()
        path_lifetime_history_time = "lifetime_history_time.pkl"
        with open(path_lifetime_history_time, "wb") as f:
            pickle.dump(self.lifetime_history_time, f)
            
        self.__del__()

    def __del__(self):
        if not self.is_alive:
            return
        
        # check every worker is terminated
        time.sleep(0.5)
        for worker in self.workers:
            if worker.is_alive():
                print(f"worker {worker.pid} is still alive")
        
        logging.debug("Dataloader: __del__ finish")
        self.is_alive = False
        clear_shm_start_with(self.instance_id)



def _worker_main(data_rootpath: str, db_names: list, 
                 tokenizer, worker_id: int, instance_id: str,
                 log_dir, log_name):
    worker = Dataloader_worker(data_rootpath, db_names, tokenizer, worker_id, instance_id, log_dir, log_name)
    worker.run()
    del worker



class Dataloader_worker:
    def __init__(self, data_rootpath: str, db_names: list, 
                 tokenizer, worker_id: int, instance_id: str,
                 log_dir, log_name):
        self.worker_id = worker_id
        self.instance_id = instance_id

        self.tokenizer = tokenizer
        self.data_rootpath = data_rootpath
        self.db_names = db_names
        self.task_finished_cnt = 0

        self.db_logger = None
        if self.worker_id == 0 and log_dir is not None and log_name is not None:
            self.db_logger = DB_logger(logger_name=log_name, log_dir=log_dir)
            logging.info(f"Dataloader_worker {self.worker_id} will log to {log_dir}/{log_name}")

        self.lmdb_envs = {}
        self.lmdb_start_idx = {}
        self.lmdb_end_idx = {}
        self.open_lmdbs(data_rootpath, db_names)

        # set the random seed for numpy, then each worker will have different random sequence
        # it is CRUCIAL to have a ideal data distribution
        np.random.seed(secrets.randbits(32))
        
        print(f"worker {self.worker_id} is starting")
        self.worker_cpp = cpp_dearth_dataloader.DearthDataloaderWorkerCpp(worker_id, instance_id)


    def open_lmdbs(self, data_rootpath, db_names):
        for name in db_names:
            db_path = os.path.join(data_rootpath, name)
            try:
                self.lmdb_envs[name] = lmdb.open(db_path, readonly=True, lock=False, readahead=False, meminit=False)
                self.lmdb_start_idx[name] = 0
                self.lmdb_end_idx[name] = self.lmdb_envs[name].stat()["entries"]
                if self.worker_id == 0:
                    logging.info(f"lmdb {name} has {self.lmdb_end_idx[name]} samples")
            except:
                logging.error(f"failed to open lmdb {name}")

    def close_lmdbs(self):
        if self.lmdb_envs is None:
            return
        for name in self.lmdb_envs:
            try:
                self.lmdb_envs[name].close()
            except:
                logging.error(f"failed to close lmdb {name}")
    
    def run(self):
        while True:
            if self.worker_cpp.poll_terminate():
                break
            
            task = self.worker_cpp.poll_task()
            if task is not None:
                task_id = task["task_id"]
                db_name = task["db_name"]
                task_num_samples = task["num_samples"]

                data, total_len = self.get_data(db_name, task_num_samples)
                self.worker_cpp.send_result(task_id, total_len, db_name, data)
                
                self.task_finished_cnt += 1
            else:
                time.sleep(0.05)


    def get_data(self, db_name, num_samples):
        """
        return: list of tokens, List[List[int]]
        """
        db_env = self.lmdb_envs[db_name]
        db_start_idx = self.lmdb_start_idx[db_name]
        db_end_idx = self.lmdb_end_idx[db_name]
        ret = []
        total_len = 0
        with db_env.begin(write=False) as txn:
            rand_idxs = np.random.randint(db_start_idx, db_end_idx, size=num_samples)
            rand_idxs = np.sort(rand_idxs)
            #print(f"worker {self.worker_id} get {num_samples} samples from {db_name}, start_idx = {rand_idxs[0]}, end_idx = {rand_idxs[-1]}, {rand_idxs.shape}")
            #print(rand_idxs[0:20])
            if DEBUG_MODE and self.db_logger is not None:
                self.db_logger.log("data_idx", rand_idxs, self.task_finished_cnt, "sampling-info")
            for i in range(num_samples):
                rand_idx = int(rand_idxs[i])
                text = txn.get(str(rand_idx).encode('utf-8'))
                if text is None:
                    continue
                text = lz4.frame.decompress(text)
                text = text.decode('utf-8')
                tokens = self.tokenizer.encode(text) # TODO: change here if tokenizer using a different tokenization method
                # ensure tokens are List[int]
                if not isinstance(tokens, list):
                    tokens = tokens.tolist()
                tokens = post_tokenize_process(tokens, self.tokenizer)
                if DEBUG_MODE and self.db_logger is not None and self.task_finished_cnt % 10 == 0 and i == 0:
                    self.db_logger.log("tokens", tokens, self.task_finished_cnt, "sampling-info")
                ret.append(tokens)
                total_len += len(tokens)
        return ret, total_len

    def __del__(self):
        if self is None:
            return
        if not hasattr(self, "worker_id"):
            return
        self.close_lmdbs()
        logging.info(f"worker {self.worker_id} is exiting")
        




def post_tokenize_process(token_ids, tokenizer) -> list:
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    ret = [bos_id] # 1 is the token_id of <s>
    # in this design, we will ensure that the first token is <s>, and the last token is </s>

    i = 0
    for token_id in token_ids:
        if i == 0 and token_id == bos_id:
            i += 1
            continue
        if token_id == bos_id: # 1 is the token_id of <s>; prevent any bos_id in the middle
            ret.append(tokenizer.get_vocab()["<"])
            ret.append(tokenizer.get_vocab()["s"])
            ret.append(tokenizer.get_vocab()[">"])
            logging.warn(f"token_id 1 appears in the middle of the text, i={i}")
        elif token_id == eos_id: # 2 is the token_id of </s>; prevent any eos_id
            ret.append(tokenizer.get_vocab()["<"])
            ret.append(tokenizer.get_vocab()["/"])
            ret.append(tokenizer.get_vocab()["s"])
            ret.append(tokenizer.get_vocab()[">"])
            logging.warn(f"token_id 2 appears in the middle of the text, i={i}")
        else:
            ret.append(token_id)
        i += 1
    
    ret.append(eos_id)
    
    return ret
