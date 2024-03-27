#ifndef CPP_DATALOADER_LIB_H
#define CPP_DATALOADER_LIB_H


// Install Boost for Ubuntu: sudo apt-get install libboost-all-dev
// the location of the header files is /usr/include/boost
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>

#include <iostream>
#include <vector>
#include <list>
#include <random>


double rand_range(double min, double max, std::mt19937 gen) {
    std::uniform_real_distribution<double> dis(min, max);
    return dis(gen);
}


// @return 0 if success, 1 if failed
int create_shared_memory(const char* shm_id, long shm_size) {
    using namespace boost::interprocess;
    try {
        // clean up the shared memory if it already exists
        shared_memory_object::remove(shm_id);

        shared_memory_object shm_obj(create_only, shm_id, read_write);
        shm_obj.truncate(shm_size);
        mapped_region region(shm_obj, read_write);
    } catch (interprocess_exception &ex) {
        std::cout << "create_shared_memory failed for " << shm_id << std::endl;
        std::cout << ex.what() << std::endl;
        return 1;
    }
    return 0;
}


template <typename T>
size_t count_2darray_nbytes(const std::vector<std::vector<T>>& data) {
    // get element count of the 2d array
    size_t num_rows = data.size();
    size_t element_cnt = 0;
    for (size_t i = 0; i < num_rows; i++) {
        element_cnt += data[i].size();
    }
    return element_cnt * sizeof(T) + sizeof(size_t) + sizeof(size_t) * num_rows;
}

void write_2darray_to_addr(const std::vector<std::vector<long>>& data, void* addr) {
    // Example: [[1, 1], [2,2,2]] -> 2 2 1 1 3 2 2 2
    // first number is the number of rows
    // then for each row, first number is the number of elements in the row

    char* start_ptr = (char*)addr;
    size_t num_rows = data.size();
    *(size_t*)start_ptr = num_rows;
    start_ptr += sizeof(size_t);

    for (size_t i = 0; i < num_rows; i++) {
        size_t current_row_size = data[i].size();
        *(size_t*)start_ptr = current_row_size;
        start_ptr += sizeof(size_t);

        // memcmp(start_ptr, &data[i][0], sizeof(long) * current_row_size);
        // start_ptr += sizeof(long) * current_row_size;

        for (size_t j = 0; j < current_row_size; j++) {
            *(long*)start_ptr = data[i][j];
            start_ptr += sizeof(long);
        }
    }
}

void read_2darray_from_addr(std::vector<std::vector<long>> &data, void* addr) {
    // Example: 2 2 1 1 3 2 2 2 -> [[1, 1], [2,2,2]]

    char* start_ptr = (char*)addr;
    size_t num_rows = *(size_t*)start_ptr;
    start_ptr += sizeof(size_t);

    data.resize(num_rows);
    for (size_t i = 0; i < num_rows; i++) {
        size_t current_row_size = *(size_t*)start_ptr;
        start_ptr += sizeof(size_t);

        data[i].resize(current_row_size);
        
        // memcmp(&data[i][0], start_ptr, sizeof(long) * current_row_size);
        // start_ptr += sizeof(long) * current_row_size;

        for (size_t j = 0; j < current_row_size; j++) {
            data[i][j] = *(long*)start_ptr;
            start_ptr += sizeof(long);
        }
    }
}

// @return 0 if success, 1 if failed
int put_2darray_to_shared_memory(const char* shm_id, const std::vector<std::vector<long>>& data) {
    using namespace boost::interprocess;

    size_t n_bytes = count_2darray_nbytes(data);

    if (create_shared_memory(shm_id, n_bytes) != 0) {
        return 1;
    }

    try {
        shared_memory_object shm_obj(open_only, shm_id, read_write);
        mapped_region region(shm_obj, read_write);
        void* addr = region.get_address();
        write_2darray_to_addr(data, addr);
    } catch (interprocess_exception &ex) {
        std::cout << "put_2darray_to_shared_memory failed for " << shm_id << std::endl;
        std::cout << ex.what() << std::endl;
        return 1;
    }
    return 0;
}

std::vector<std::vector<long>> get_2darray_from_shared_memory(const char* shm_id) {
    using namespace boost::interprocess;

    std::vector<std::vector<long>> data;

    try {
        shared_memory_object shm_obj(open_only, shm_id, read_only);
        mapped_region region(shm_obj, read_only);
        void* addr = region.get_address();
        read_2darray_from_addr(data, addr);
    } catch (interprocess_exception &ex) {
        std::cout << "get_2darray_from_shared_memory failed for " << shm_id << std::endl;
        std::cout << ex.what() << std::endl;
    }
    return data;
}

int get_2darray_from_shared_memory(const char* shm_id, std::vector<std::vector<long>>& data) {
    using namespace boost::interprocess;

    try {
        shared_memory_object shm_obj(open_only, shm_id, read_only);
        mapped_region region(shm_obj, read_only);
        void* addr = region.get_address();
        read_2darray_from_addr(data, addr);
    } catch (interprocess_exception &ex) {
        std::cout << "get_2darray_from_shared_memory failed for " << shm_id << std::endl;
        std::cout << ex.what() << std::endl;
        return 1;
    }
    return 0;
}


void delete_shared_memory(const char* shm_id) {
    using namespace boost::interprocess;
    try {
        shared_memory_object::remove(shm_id);
    } catch (interprocess_exception &ex) {
        std::cout << "delete_shared_memory failed for " << shm_id << std::endl;
        std::cout << ex.what() << std::endl;
    }
}


unsigned long get_time_us() {
    return std::chrono::steady_clock::now().time_since_epoch().count() / 1000;
}


struct SampleBufferSlowRelease {
    double MAX_SLOW_RELEASE_RATE = 0.4;
    size_t MAX_SLOW_RELEASE_BUF_SIZE = 2500;
    size_t DIFFERENCE_TO_ALIGN = 25; // if the nearest segment starting point is within this range, then align to it
    
    const size_t MAX_SEGMENT_OUT_FROM_TMP_BUF = 5;

    std::vector<long> tmp_buf;
    std::vector<std::vector<long>> slow_release_buf; // with length = recent_num_tokens
    std::vector<std::vector<long>> ready_buffer; // with length = recent_num_tokens
    //std::vector<std::vector<long>> sample_buffer;
    std::list<std::vector<long>> sample_buffer;

    size_t sample_buffer_size = 0;

    long avg_stats_total_len = 0;
    long avg_stats_row_cnt = 0;

    unsigned long speed_stats_prev_time = 0;
    size_t speed_stats_total_tokens = 0.0;
    double speed_stats_total_time = 0.1;

    size_t recent_num_tokens = 0;

    std::mt19937 gen;

    SampleBufferSlowRelease() {
        gen = std::mt19937(std::random_device()());
    }

    SampleBufferSlowRelease(const SampleBufferSlowRelease&) {
        gen = std::mt19937(std::random_device()());
    }

    SampleBufferSlowRelease& operator=(const SampleBufferSlowRelease&) {
        gen = std::mt19937(std::random_device()());
        return *this;
    }

    size_t get_buf_size() { return sample_buffer_size; }
    size_t get_buf_num_rows() { return sample_buffer.size(); }

    double get_consuming_speed() {
        return this->speed_stats_total_time == 0 ? 0.0 : 
            this->speed_stats_total_tokens / this->speed_stats_total_time;
    }
    double get_avg_token_len() {
        return this->avg_stats_row_cnt == 0 ? 400.0 : 
            this->avg_stats_total_len / this->avg_stats_row_cnt;
    }

    bool enough_for(size_t num_tokens) {
        if (this->get_buf_size() >= num_tokens) {
            return true;
        }
        if (this->recent_num_tokens == num_tokens && this->ready_buffer.size() > 0) {
            return true;
        }
        return false;
    }


    void extend(std::vector<std::vector<long>> list_of_tokens, size_t total_num_tokens) {
        this->avg_stats_total_len += total_num_tokens;
        this->avg_stats_row_cnt += list_of_tokens.size();
        if (this->avg_stats_row_cnt > 20000) {
            this->avg_stats_total_len /= 2;
            this->avg_stats_row_cnt /= 2;
        }

        this->sample_buffer_size += total_num_tokens;
        this->sample_buffer.insert(this->sample_buffer.end(), list_of_tokens.begin(), list_of_tokens.end());
    }

    void _report_sample_speed(size_t num_tokens) {
        if (this->speed_stats_prev_time == 0) {
            this->speed_stats_prev_time = get_time_us();
            return;
        }

        this->speed_stats_total_tokens += num_tokens;
        auto current_time = get_time_us();
        this->speed_stats_total_time += (current_time - this->speed_stats_prev_time) / 1e6; // in seconds
        this->speed_stats_prev_time = current_time;

        if (this->speed_stats_total_time > 100.0) {
            this->speed_stats_total_tokens /= 2;
            this->speed_stats_total_time /= 2;
        }
    }

    // @return 0 if success, 1 if failed
    int sample(size_t num_tokens, std::vector<std::unique_ptr<long[]>>& result) {
        if (this->sample_buffer_size < num_tokens || num_tokens <= 0) {
            return 1;
        }

        if (this->recent_num_tokens != num_tokens) {
            this->ready_buffer.clear();
            this->slow_release_buf.clear();
            this->recent_num_tokens = num_tokens;
        } else {
            std::bernoulli_distribution dist(
                std::max(0.0001, this->slow_release_buf.size() / this->MAX_SLOW_RELEASE_BUF_SIZE * this->MAX_SLOW_RELEASE_RATE)
            );
            if (this->slow_release_buf.size() > 0 && dist(this->gen)) {
                this->_report_sample_speed(num_tokens);
                //result.push_back(this->slow_release_buf.back());
                auto new_sample = std::make_unique<long[]>(num_tokens);
                std::copy(this->slow_release_buf.back().begin(), this->slow_release_buf.back().end(), new_sample.get());
                result.push_back(std::move(new_sample));
                this->slow_release_buf.pop_back();
                return 0;
            }

            if (this->ready_buffer.size() > 0) {
                this->_report_sample_speed(num_tokens);
                //result.push_back(this->ready_buffer.back());
                auto new_sample = std::make_unique<long[]>(num_tokens);
                std::copy(this->ready_buffer.back().begin(), this->ready_buffer.back().end(), new_sample.get());
                result.push_back(std::move(new_sample));
                this->ready_buffer.pop_back();
                return 0;
            }
        }

        // there is no ready tokens in ready_buffer
        while (this->sample_buffer.size() > 0 && this->tmp_buf.size() < recent_num_tokens) {
            this->sample_buffer_size -= this->sample_buffer.back().size();
            this->tmp_buf.insert(this->tmp_buf.end(), this->sample_buffer.back().begin(), this->sample_buffer.back().end());
            this->sample_buffer.pop_back();
        }

        if (this->tmp_buf.size() < recent_num_tokens) {
            return 1;
        }

        // get segment from tmp_buf
        std::vector<std::vector<long>> seg;
        size_t start_idx = 0;
        while (start_idx + num_tokens <= this->tmp_buf.size()) {
            seg.push_back(std::vector<long>(this->tmp_buf.begin() + start_idx, this->tmp_buf.begin() + start_idx + num_tokens));
            start_idx += num_tokens;
        }
        this->tmp_buf.erase(this->tmp_buf.begin(), this->tmp_buf.begin() + start_idx);
        
        //result.push_back(seg[0]);
        auto new_sample = std::make_unique<long[]>(num_tokens);
        std::copy(seg[0].begin(), seg[0].end(), new_sample.get());
        result.push_back(std::move(new_sample));

        seg.erase(seg.begin());
        
        // this function prefers the sample start with <s>, unless the sample is extremely long;
        // it may make the distribution more similar to the inference input, 
        // because the inference will always start with <s>, as the first token
        if (this->tmp_buf.size() < this->DIFFERENCE_TO_ALIGN) {
            this->tmp_buf.clear();
        }
        
        // there is 3 conditions: 1. only one seg; 2. #seg < MAX_SEGMENT_OUT_FROM_TMP_BUF; 3. #seg >= MAX_SEGMENT_OUT_FROM_TMP_BUF, which need to put into slow_release_buffer
        if (seg.size() == 0) {
            this->_report_sample_speed(num_tokens);
            return 0;
        }
        if (seg.size() < this->MAX_SEGMENT_OUT_FROM_TMP_BUF) {
            this->ready_buffer.insert(this->ready_buffer.end(), seg.begin(), seg.end());
        } else {
            this->ready_buffer.insert(this->ready_buffer.end(), seg.begin(), seg.begin() + this->MAX_SEGMENT_OUT_FROM_TMP_BUF);
            seg.erase(seg.begin(), seg.begin() + this->MAX_SEGMENT_OUT_FROM_TMP_BUF);

            this->slow_release_buf.insert(this->slow_release_buf.end(), seg.begin(), seg.end());

            // suffle the slow_release_buf
            std::shuffle(this->slow_release_buf.begin(), this->slow_release_buf.end(), this->gen);

            if (this->slow_release_buf.size() > this->MAX_SLOW_RELEASE_BUF_SIZE) {
                this->slow_release_buf.erase(this->slow_release_buf.begin(), 
                    this->slow_release_buf.begin() + this->slow_release_buf.size() - this->MAX_SLOW_RELEASE_BUF_SIZE);
            }
        }

        this->_report_sample_speed(num_tokens);
        return 0;
    }
};


#endif // CPP_DATALOADER_LIB_H