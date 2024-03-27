#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <random>
#include <map>
#include <chrono>

#include "cpp_dataloader_lib.h"

#include <boost/interprocess/ipc/message_queue.hpp>
#include "boost/date_time/posix_time/posix_time_types.hpp"


namespace py = pybind11;

int py_put_list_tokens_to_shared_memory(py::str py_shm_id, py::list list_tokens) {
    std::vector<std::vector<long>> data = py::cast<std::vector<std::vector<long>>>(list_tokens);
    std::string cpp_str_shm_id = py::cast<std::string>(py_shm_id);
    return put_2darray_to_shared_memory(cpp_str_shm_id.c_str(), data);
}

int put_list_tokens_to_shared_memory(std::string shm_id, py::list list_tokens) {
    std::vector<std::vector<long>> data = py::cast<std::vector<std::vector<long>>>(list_tokens);
    return put_2darray_to_shared_memory(shm_id.c_str(), data);
}

py::object py_get_list_tokens_from_shared_memory(py::str py_shm_id) {
    std::string cpp_str_shm_id = py::cast<std::string>(py_shm_id);
    std::vector<std::vector<long>> data;
    int status = get_2darray_from_shared_memory(cpp_str_shm_id.c_str(), data);
    if (status != 0) {
        return py::none();
    }
    return py::cast(data);
}

void py_delete_shared_memory(py::str py_shm_id) {
    std::string cpp_str_shm_id = py::cast<std::string>(py_shm_id);
    delete_shared_memory(cpp_str_shm_id.c_str());
}


std::string num_to_base36(long num) {
    if (num < 0) {
        return num_to_base36(-num);
    }
    std::string base36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::string ret = "";
    if (num == 0) {
        return "0";
    }
    while (num > 0) {
        ret = base36[num % 36] + ret;
        num /= 36;
    }
    return ret;
}


#define SHM_PREFIX "dearthdl_"

std::string get_shm_prefix() {
    return SHM_PREFIX;
}

std::string get_output_queue_name(std::string instance_id) {
    return std::string(SHM_PREFIX) + instance_id + "_output_queue";
}

std::string get_task_queue_name(std::string instance_id) {
    return std::string(SHM_PREFIX) + instance_id + "_task_queue";
}

std::string get_terminate_queue_name(std::string instance_id) {
    return std::string(SHM_PREFIX) + instance_id + "_terminate_queue";
}

std::string get_result_shm_name(std::string instance_id, int worker_id, unsigned long task_id) {
    return std::string(SHM_PREFIX) + instance_id + "_" + num_to_base36(worker_id) + "_" + num_to_base36(task_id);
}

std::string to_shm_path(std::string name) {
    return "/dev/shm/" + name;
}




#define MAX_QUEUE_SIZE 262144
#define MAX_TERMINATE_QUEUE_SIZE 512
#define SQ(T,N) es::lockfree::shared_mpmc_queue<es::lockfree::mpmc_queue<T, N>>
#define MQ boost::interprocess::message_queue

using std::vector;


struct WorkerTask {
    unsigned long task_id;
    unsigned long num_samples;
    char db_name[128-16];
};

struct WorkerResult {
    unsigned long task_id;
    unsigned long total_len;
    char db_name[128-8];
    char result_shm_name[128-8];
};


struct DearthDataloaderWorkerCpp {
    MQ* output_queue;
    MQ* task_queue;
    MQ* terminate_queue;
    
    int worker_id;
    std::string instance_id;

    DearthDataloaderWorkerCpp(int worker_id, std::string instance_id) {
        this->worker_id = worker_id;
        this->instance_id = std::string(instance_id);


        this->output_queue = new MQ(
            boost::interprocess::open_or_create, 
            get_output_queue_name(this->instance_id).c_str(),
            MAX_QUEUE_SIZE,
            sizeof(WorkerResult)
        );

        this->task_queue = new MQ(
            boost::interprocess::open_or_create, 
            get_task_queue_name(this->instance_id).c_str(),
            MAX_QUEUE_SIZE,
            sizeof(WorkerTask)
        );


        this->terminate_queue = new MQ(
            boost::interprocess::open_or_create, 
            get_terminate_queue_name(this->instance_id).c_str(),
            MAX_TERMINATE_QUEUE_SIZE,
            sizeof(int)
        );
    }

    ~DearthDataloaderWorkerCpp() {
        delete this->output_queue;
        delete this->task_queue;
        delete this->terminate_queue;
    }


    py::object py_poll_task() {
        WorkerTask task;
        size_t received_size = 0;
        unsigned int received_priority = 0;
        // max wait time: 10us
        boost::posix_time::ptime max_timeout = boost::posix_time::microsec_clock::universal_time() + boost::posix_time::microseconds(10);
        bool status = this->task_queue[0].timed_receive(&task, sizeof(task), received_size, received_priority, max_timeout);
        if (!status) {
            return py::none();
        } else {
            py::dict result;
            result["task_id"] = task.task_id;
            result["db_name"] = std::string(task.db_name);
            result["num_samples"] = task.num_samples;
            return result;
        }
    }

    void py_send_result(unsigned long task_id, long total_len, py::str db_name, py::list tokens_list) {
        if (total_len <= 0) {
            send_result(task_id, 0, py::cast<std::string>(db_name), "");
            return;
        }

        auto shm_name = get_result_shm_name(this->instance_id, this->worker_id, task_id);
        int status = put_list_tokens_to_shared_memory(shm_name, tokens_list);
        if (status != 0) {
            return; // do nothing if shm failed; will not redo the task but wait the task to expire for manager-side. 
        }
        send_result(task_id, total_len, py::cast<std::string>(db_name), shm_name);
    }

    void send_result(unsigned long task_id, long total_len, std::string db_name, std::string result_shm_name) {
        WorkerResult result;
        result.task_id = task_id;
        result.total_len = total_len;
        strncpy(result.db_name, db_name.c_str(), 100);
        strncpy(result.result_shm_name, result_shm_name.c_str(), 100);
        // max_timeout: 10us
        boost::posix_time::ptime max_timeout = boost::posix_time::microsec_clock::universal_time() + boost::posix_time::microseconds(10);
        this->output_queue[0].timed_send(&result, sizeof(result), 0, max_timeout);
    }

    py::bool_ py_poll_terminate() {
        int terminate_signal;
        size_t received_size = 0;
        unsigned int received_priority = 0;
        // max wait time: 0us
        boost::posix_time::ptime max_timeout = boost::posix_time::microsec_clock::universal_time() + boost::posix_time::microseconds(0);
        bool status = this->terminate_queue[0].timed_receive(&terminate_signal, sizeof(terminate_signal), received_size, received_priority, max_timeout);
        return py::bool_(status);
    }
    
};



struct DearthDataloaderManagerCpp {
    MQ* output_queue;
    MQ* task_queue;
    MQ* terminate_queue;

    vector<std::thread> manager_threads; // in this version, only one manager thread because many objects do not have locks yet. 

    std::map<size_t, std::list<std::unique_ptr<long[]>>> batch_tokens_buf; // seq_len -> [tokens]
    std::map<size_t, std::tuple<unsigned long, std::string, size_t>> waiting_tasks;
    std::map<std::string, SampleBufferSlowRelease> raw_tokens_each_db;

    std::mutex batch_tokens_buf_lock;
    std::mutex recent_seqlen_lock;
    std::mutex exit_lock;

    size_t recent_seqlen = 0;
    size_t recent_batch_size = 1000;
    size_t cnt_tasks_sent = 0;

    vector<std::string> db_names;
    vector<double> db_weights;
    size_t worker_cnt;
    std::string instance_id;

    std::mt19937 gen;

    unsigned long batch_start_time = 0;
    long batch_id = 0;

    bool terminate = false;

    DearthDataloaderManagerCpp(vector<std::string> db_names, vector<double> db_weights, long worker_cnt, std::string instance_id) {
        assert(db_names.size() == db_weights.size());
        assert(worker_cnt > 0);

        this->db_names = vector<std::string>(db_names);
        this->db_weights = vector<double>(db_weights);

        this->worker_cnt = worker_cnt;
        this->instance_id = std::string(instance_id);

        this->gen = std::mt19937(std::random_device()());
        this->terminate = false;

        this->output_queue = new MQ(
            boost::interprocess::open_or_create, 
            get_output_queue_name(this->instance_id).c_str(),
            MAX_QUEUE_SIZE,
            sizeof(WorkerResult)
        );

        this->task_queue = new MQ(
            boost::interprocess::open_or_create, 
            get_task_queue_name(this->instance_id).c_str(),
            MAX_QUEUE_SIZE,
            sizeof(WorkerTask)
        );

        this->terminate_queue = new MQ(
            boost::interprocess::open_or_create, 
            get_terminate_queue_name(this->instance_id).c_str(),
            MAX_TERMINATE_QUEUE_SIZE,
            sizeof(int)
        );


        // start the manager loop thread
        std::thread manager_thread(&DearthDataloaderManagerCpp::manager_run, this);
        manager_thread.detach();
        this->manager_threads.push_back(std::move(manager_thread));
    }

    void destory() {
        // notify workers to terminate
        for (size_t i = 0; i < this->worker_cnt*2; i++) {
            boost::posix_time::ptime max_timeout = boost::posix_time::microsec_clock::universal_time() + boost::posix_time::microseconds(100);
            int cmd = 1;
            this->terminate_queue[0].timed_send(&cmd, sizeof(cmd), 0, max_timeout);
        }
        {
            std::lock_guard<std::mutex> lock(this->exit_lock);
            this->terminate = true;
        }
    }

    ~DearthDataloaderManagerCpp() {
        delete this->output_queue;
        delete this->task_queue;
        delete this->terminate_queue;
        std::cout << "DearthDataloaderManagerCpp destructed" << std::endl;
    }

    // py::array_t<long>
    py::object get_batch(int batch_size, int seqlen) {
        if (this->batch_start_time == 0) {
            this->batch_start_time = get_time_us();
        }

        auto _ = this->set_bz_seqlen(batch_size, seqlen);
        batch_size = std::get<0>(_);
        seqlen = std::get<1>(_);

        if (!this->has_enough_for_batch(batch_size, seqlen)) {
            bool flag_not_enough = true;
            for (int i = 0; i < 50; i++) {
                std::this_thread::sleep_for(std::chrono::microseconds(400));
                if (this->has_enough_for_batch(batch_size, seqlen)) {
                    flag_not_enough = false;
                    break;
                }
            }
            if (flag_not_enough) {
                return py::none();
            }
        }

        py::array_t<long> result({batch_size, seqlen});

        std::lock_guard<std::mutex> lock(this->batch_tokens_buf_lock);

        auto& batch_buf = this->batch_tokens_buf[seqlen];
        auto batch_tokens_buf_iter = batch_buf.begin();
        // copy to numpy array
        auto raw_arr = result.mutable_unchecked<2>();
        auto begin_raw_arr_ptr = raw_arr.mutable_data(0, 0);
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < seqlen; j++) {
                //*raw_arr.mutable_data(i, j) = tmp_cpy_buf[i*seqlen + j];
                *begin_raw_arr_ptr = (*batch_tokens_buf_iter)[j];
                //(*batch_tokens_buf_iter)[j];
                begin_raw_arr_ptr++;
            }
            batch_tokens_buf_iter++;
        }

        // remove from buffer
        //batch_buf.erase(batch_buf.begin(), batch_buf.begin() + batch_size);
        for (int i = 0; i < batch_size; i++) {
            batch_buf.pop_front();
        }

        this->batch_id++;
        this->batch_start_time = 0;

        return result;
    }

    long get_raw_buffer_token_size() {
        long total_size = 0;
        for (auto& kv : this->raw_tokens_each_db) {
            total_size += kv.second.get_buf_size();
        }
        return total_size;
    }

    long get_raw_buffer_row_size() {
        long total_size = 0;
        for (auto& kv : this->raw_tokens_each_db) {
            total_size += kv.second.get_buf_num_rows();
        }
        return total_size;
    }

    long get_ready_buffer_row_size(int seqlen) {
        std::lock_guard<std::mutex> lock(this->batch_tokens_buf_lock);
        if (this->batch_tokens_buf.find(seqlen) == this->batch_tokens_buf.end()) {
            return 0;
        }
        return this->batch_tokens_buf[seqlen].size();
    }


    bool has_enough_for_batch(int batch_size, int seqlen) {
        std::lock_guard<std::mutex> lock(this->batch_tokens_buf_lock);
        if (this->batch_tokens_buf.find(seqlen) == this->batch_tokens_buf.end()) {
            return false;
        }
        return this->batch_tokens_buf[seqlen].size() >= (size_t)batch_size;
    }

    std::tuple<int, int> set_bz_seqlen(int batch_size, int seqlen) {
        std::lock_guard<std::mutex> lock(this->recent_seqlen_lock);
        if (batch_size > 0) {
            this->recent_batch_size = batch_size;
        }
        if (seqlen > 0) {
            this->recent_seqlen = seqlen;
        }
        return std::tuple<int, int>(this->recent_batch_size, this->recent_seqlen);
    }


    void manager_run() {
        auto manager_start_time = get_time_us();
        while (true) {
            long tmp_status = dump();
            tmp_status += fill_buffer();
            tmp_status += fetch_more();
            if (tmp_status > 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(3000));
            }

            std::lock_guard<std::mutex> lock(this->exit_lock);
            if (this->terminate) {
                std::cout << "manager_run: terminate" << std::endl;
                break;
            }

            auto tmp_time = get_time_us();
            if (tmp_time - manager_start_time > 10000) {
                manager_start_time = tmp_time;
            }

        }
    }

    long dump() {
        int MAX_DUMP_CNT = 1000; // max number of tasks to dump
        unsigned long MAX_DUMP_TIME = 10000; // 0.02s, in microseconds
        unsigned long start_dump_time = get_time_us();

        unsigned long sample_total = 0;
        long task_cnt = 0;

        for (int i = 0; i < MAX_DUMP_CNT; i++) {
            if (get_time_us() - start_dump_time > MAX_DUMP_TIME) {
                break;
            }
            WorkerResult result;
            boost::posix_time::ptime max_timeout = boost::posix_time::microsec_clock::universal_time() + boost::posix_time::microseconds(10);
            size_t recved_size = 0;
            unsigned int recved_priority = 0;
            bool status = this->output_queue[0].timed_receive(&result, sizeof(result), recved_size, recved_priority, max_timeout);

            if (!status) {
                break;
            }
            task_cnt++;

            // remove from task waiting list
            if (this->waiting_tasks.find(result.task_id) != this->waiting_tasks.end()) {
                this->waiting_tasks.erase(result.task_id);
            }

            std::string empty_str = "";
            if (result.total_len > 0 && result.result_shm_name != empty_str) {
                vector<vector<long>> new_tokens_list;
                auto status = get_2darray_from_shared_memory(result.result_shm_name, new_tokens_list);
                if (status != 0) {
                    delete_shared_memory(result.result_shm_name);
                    continue;
                }

                if (new_tokens_list.size() <= 0) {
                    continue;
                }
                
                if (this->raw_tokens_each_db.find(result.db_name) == this->raw_tokens_each_db.end()) {
                    this->raw_tokens_each_db[result.db_name] = SampleBufferSlowRelease();
                }
                this->raw_tokens_each_db[result.db_name].extend(new_tokens_list, result.total_len);
                sample_total += new_tokens_list.size();
            }
            
            // remove shared memory
            if (result.result_shm_name != empty_str) {
                delete_shared_memory(result.result_shm_name);
            }
        }

        return task_cnt;
    }

    long fill_buffer() {
        size_t buf_size = 0;
        size_t required_seqlen = 0;
        {
            std::lock_guard<std::mutex> lock(this->recent_seqlen_lock);
            buf_size = this->recent_batch_size;
            required_seqlen = this->recent_seqlen;
        }


        if (required_seqlen == 0) {
            return 0;
        }

        auto start_time = get_time_us();
        constexpr size_t MAX_FILL_TIME = 20000; // 0.02s = 20000us
        constexpr size_t MIN_BUF_SIZE = 12000;
        buf_size = std::max(required_seqlen * 1, MIN_BUF_SIZE);

        size_t current_buf_size = 0;
        {
            std::lock_guard<std::mutex> lock(this->batch_tokens_buf_lock);
            if (this->batch_tokens_buf.find(required_seqlen) == this->batch_tokens_buf.end()) {
                this->batch_tokens_buf[required_seqlen] = std::list<std::unique_ptr<long[]>>();
            }
            current_buf_size = this->batch_tokens_buf[required_seqlen].size();
        }

        if (current_buf_size >= buf_size) {
            return 0;
        }

        std::discrete_distribution<size_t> db_dist(this->db_weights.begin(), this->db_weights.end());
        
        vector<std::unique_ptr<long[]>> new_tokens_list;
        while (new_tokens_list.size() + current_buf_size < buf_size) {
            // ensure every dataset has been fetched, and every dataset has at least one sample with required_seqlen
            for (auto& db_name : this->db_names) {
                if (this->raw_tokens_each_db.find(db_name) == this->raw_tokens_each_db.end()) {
                    goto END_OF_FILL_BUFFER;
                }
                if (!this->raw_tokens_each_db[db_name].enough_for(required_seqlen)) {
                    goto END_OF_FILL_BUFFER;
                }
            }

            auto db_name_idx = db_dist(this->gen);
            auto db_name = this->db_names[db_name_idx];
            auto& raw_buf = this->raw_tokens_each_db[db_name];
            auto status = raw_buf.sample(required_seqlen, new_tokens_list);
            if (status != 0) {
                goto END_OF_FILL_BUFFER;
            }
            if (new_tokens_list.size() % 100 == 99 && get_time_us() - start_time > MAX_FILL_TIME) {
                goto END_OF_FILL_BUFFER;
            }
        }
        
        END_OF_FILL_BUFFER:
        if (new_tokens_list.size() > 0) {
            std::lock_guard<std::mutex> lock(this->batch_tokens_buf_lock);
            //this->batch_tokens_buf[required_seqlen].insert(this->batch_tokens_buf[required_seqlen].end(), new_tokens_list.begin(), new_tokens_list.end());
            for (auto& tokens : new_tokens_list) {
                this->batch_tokens_buf[required_seqlen].push_back(std::move(tokens));
            }
        }

        return new_tokens_list.size();
    }

    long fetch_more() {
        auto current_time = get_time_us();
        std::map<std::string, size_t> assigned_sample_num;
        
        size_t cnt_waiting_tasks = 0;
        vector<size_t> task_ids_to_remove;
        for (auto& kv : this->waiting_tasks) {
            auto task_id = kv.first;
            auto task_info = kv.second;
            auto task_assign_time = std::get<0>(task_info);
            auto db_name = std::get<1>(task_info);
            auto num_samples = std::get<2>(task_info);

            if (current_time - task_assign_time > 30 * 1000000) { // if task is assigned for more than 30s, remove it
                task_ids_to_remove.push_back(task_id);
                continue;
            }

            cnt_waiting_tasks++;
            if (assigned_sample_num.find(db_name) == assigned_sample_num.end()) {
                assigned_sample_num[db_name] = 0;
            }
            assigned_sample_num[db_name] += num_samples;
        }

        // remove tasks that are too old
        for (auto task_id : task_ids_to_remove) {
            this->waiting_tasks.erase(task_id);
        }

        // if there are too many waiting tasks, we don't fetch more data
        if (cnt_waiting_tasks > 60) {
            return 0;
        }

        constexpr size_t MAX_SIMPLE_EACH_TASK = 4000;
        constexpr size_t MIN_SIMPLE_EACH_TASK = 2;

        constexpr size_t MIN_ROW_EACH_BUF = 50;
        constexpr size_t MIN_TOKENS_EACH_BUF = 1 * 1e6;

        constexpr size_t  INIT_SIMPLE_EACH_DB = 200;

        // schedule the number of samples to fetch
        std::map<std::string, std::tuple<long, long>> samples_to_fetch_each_buf;
        for (size_t i = 0; i < this->db_names.size(); i++) {
            auto db_name = this->db_names[i];
            if (this->raw_tokens_each_db.find(db_name) == this->raw_tokens_each_db.end()) { // if db not in raw_tokens_each_db, never received from worker
                if (assigned_sample_num.find(db_name) == assigned_sample_num.end()) // if db never assigned
                    samples_to_fetch_each_buf[db_name] = std::tuple<size_t, size_t>(INIT_SIMPLE_EACH_DB, INIT_SIMPLE_EACH_DB/4);
                continue;
            }

            auto& raw_buf = this->raw_tokens_each_db[db_name];
            long predicted_consumption_next_8s = raw_buf.get_consuming_speed() * 8;
            predicted_consumption_next_8s = std::max(predicted_consumption_next_8s, (long)MIN_TOKENS_EACH_BUF);

            if ((long)(raw_buf.get_buf_size() + 50000) >= predicted_consumption_next_8s) {
                continue; // if the buffer is enough for next n seconds, do nothing
            }

            long target_tokens_cnt = predicted_consumption_next_8s + std::max<long>(8e6, predicted_consumption_next_8s * 0.08);
            long shortage_num_tokens = target_tokens_cnt - raw_buf.get_buf_size();

            double avg_len = raw_buf.get_avg_token_len();
            avg_len = std::max(avg_len, 10.0);

            if (assigned_sample_num.find(db_name) != assigned_sample_num.end()) {
                shortage_num_tokens -= assigned_sample_num[db_name] * avg_len;
            }
            if (shortage_num_tokens < 50000) {
                continue;
            }

            long shortage_row_cnt = shortage_num_tokens / avg_len;
            shortage_row_cnt = std::max<long>(shortage_row_cnt, MIN_ROW_EACH_BUF - raw_buf.get_buf_num_rows());
            shortage_row_cnt = std::max<long>(shortage_row_cnt, MIN_SIMPLE_EACH_TASK);

            long max_each_task = std::max<long>(MIN_SIMPLE_EACH_TASK, 2e6/avg_len);

            samples_to_fetch_each_buf[db_name] = std::tuple<long, long>(shortage_row_cnt, max_each_task);
        }

        // split to smaller tasks for each db
        vector<std::tuple<std::string, size_t>> small_tasks;
        for (auto& kv : samples_to_fetch_each_buf) {
            auto db_name = kv.first;
            auto shortage_row_cnt = std::get<0>(kv.second);
            auto max_each_task = std::get<1>(kv.second);

            while (shortage_row_cnt > 0) {
                long num_samples = max_each_task * rand_range(0.7, 1.3, this->gen);//max_each_task +- 30%
                num_samples = std::min<long>(num_samples, shortage_row_cnt);
                num_samples = std::max<long>(num_samples, MIN_SIMPLE_EACH_TASK);
                num_samples = std::min<long>(num_samples, MAX_SIMPLE_EACH_TASK);
                shortage_row_cnt -= num_samples;
                auto tmp = std::tuple<std::string, size_t>(db_name, num_samples);
                small_tasks.push_back(tmp);
            }

        }

        // shuffle the tasks
        std::shuffle(small_tasks.begin(), small_tasks.end(), this->gen);
        long new_task_cnt = small_tasks.size();

        // assign the tasks
        for (auto& task : small_tasks) {
            auto db_name = std::get<0>(task);
            auto num_samples = std::get<1>(task);
            this->assign_task(db_name, num_samples);
        }

        return new_task_cnt;
    }

    void assign_task(std::string db_name, size_t num_samples) {
        WorkerTask task;
        size_t task_id = this->cnt_tasks_sent;
        this->cnt_tasks_sent++;
        task.task_id = task_id;
        strncpy(task.db_name, db_name.c_str(), 100);
        task.num_samples = num_samples;

        this->waiting_tasks[task_id] = std::tuple<unsigned long, std::string, size_t>(get_time_us(), db_name, num_samples);
        
        // max_timeout: 10us
        boost::posix_time::ptime max_timeout = boost::posix_time::microsec_clock::universal_time() + boost::posix_time::microseconds(10);
        this->task_queue[0].timed_send(&task, sizeof(task), 0, max_timeout);
    }
};


PYBIND11_MODULE(cpp_dearth_dataloader, m) {

    py::class_<DearthDataloaderWorkerCpp>(m, "DearthDataloaderWorkerCpp")
        .def(py::init<int, std::string>())
        .def("poll_task", &DearthDataloaderWorkerCpp::py_poll_task)
        .def("send_result", &DearthDataloaderWorkerCpp::py_send_result)
        .def("poll_terminate", &DearthDataloaderWorkerCpp::py_poll_terminate);

    py::class_<DearthDataloaderManagerCpp>(m, "DearthDataloaderManagerCpp")
        .def(py::init<vector<std::string>, vector<double>, long, std::string>())
        .def("get_batch", &DearthDataloaderManagerCpp::get_batch, py::return_value_policy::take_ownership)
        .def("destory", &DearthDataloaderManagerCpp::destory)
        .def("get_raw_buffer_token_size", &DearthDataloaderManagerCpp::get_raw_buffer_token_size)
        .def("get_raw_buffer_row_size", &DearthDataloaderManagerCpp::get_raw_buffer_row_size)
        .def("get_ready_buffer_row_size", &DearthDataloaderManagerCpp::get_ready_buffer_row_size);

    m.def("put_list_tokens_to_shared_memory", &py_put_list_tokens_to_shared_memory, "put list[list[int]] to shared memory");
    m.def("get_list_tokens_from_shared_memory", &py_get_list_tokens_from_shared_memory, "get list[list[int]] from shared memory");
    m.def("delete_shared_memory", &py_delete_shared_memory, "delete shared memory");

    m.def("get_shm_prefix", &get_shm_prefix, "get shared memory prefix");
}