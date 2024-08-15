// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/utils.h"
#include "utils.h"

#include "simd/distances_onednn.h"

#include <sys/time.h>
#include <thread>
#include <vector>

#define MAX_LOOP 10
TEST_CASE("Test Brute Force", "[float vector fp32]") {
    using Catch::Approx;

    const int64_t nb = 2000000;
    const int64_t nq = 100;
    const int64_t dim = 512;
    const int64_t k = 100;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::IP );

    const auto train_ds = GenDataSet(nb, dim);
    const auto query_ds = CopyDataSet(train_ds, nq);

    const knowhere::Json conf = {
        {knowhere::meta::DIM, dim},
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, k},
        {knowhere::meta::RADIUS, knowhere::IsMetricType(metric, knowhere::metric::IP) ? 10.0 : 0.99},
    };

    SECTION("Test Search Batch") {
//     faiss::BaseData::getState().store(faiss::BASE_DATA_STATE::MODIFIED);
     struct timeval t1,t2;
     double timeuse;
//     gettimeofday(&t1, NULL);

	 std::vector<std::function<std::vector<uint8_t>(size_t, size_t)>> gen_bitset_funcs = {
		 GenerateBitsetWithFirstTbitsSet, GenerateBitsetWithRandomTbitsSet};
	 const auto bitset_percentages = {0.1f, 0.5f, 0.9f};
	 //const auto bitset_percentages = {0.5f};
	 for (const float percentage : bitset_percentages) {
		 std::cout << "percentage ----------------------------- percentage:" << percentage << std::endl;
		 for (const auto& gen_func : gen_bitset_funcs) {
			 auto bitset_data = gen_func(nb, percentage * nb);
			 knowhere::BitsetView bitset(bitset_data.data(), nb);

			 for (int i = 0; i < MAX_LOOP; i++)
			 {
				 gettimeofday(&t1, NULL);

				 //    threads.emplace_back(WrapSearch, queryvar1);
				 auto res = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, conf, bitset);
				 gettimeofday(&t2,NULL);
				 timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
				 std::cout << "ut elpased: " << timeuse << std::endl;
			 }

		 }
	 }

//	 gettimeofday(&t2, NULL);
//	 timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;

	 std::cout << "All thread finished." << std::endl;

	}
}

