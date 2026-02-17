#ifndef SPATIAL_HASHING_H
#define SPATIAL_HASHING_H
#include "curl_slam/AABB_tree/AABB.h"
#include "curl_slam/curl_tools_light.h"
#include "curl_slam/map_attributes.h"
// #include <boost/archive/text_iarchive.hpp>
// #include <boost/archive/text_oarchive.hpp>
#include <memory>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

struct GridKey {
    int x;
    int y;
    int z;
};

struct GridKeyHash {
    std::size_t operator()(const GridKey &key) const noexcept {
        std::size_t h = 0;
        h ^= std::hash<int>{}(key.x) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct GridKeyEqual {
    bool operator()(const GridKey &a, const GridKey &b) const noexcept {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    }
};

struct GridBounds {
    int x_min;
    int x_max;
    int y_min;
    int y_max;
    int z_min;
    int z_max;
    bool operator==(const GridBounds &other) const noexcept {
        return x_min == other.x_min && x_max == other.x_max && y_min == other.y_min && y_max == other.y_max &&
               z_min == other.z_min && z_max == other.z_max;
    }
    bool operator!=(const GridBounds &other) const noexcept { return !(*this == other); }
};

// Tested for the correctness

template <typename BasicType> class SpatialHashing {
  public:
    SpatialHashing() = default;
    ~SpatialHashing() = default;
    // seg_skin_thickness and ground_skin_thickness need to be added into patchinfo
    SpatialHashing(double _voxel_length) : voxel_length(_voxel_length) { patch_key = 0; }

    struct PatchBucket {
        std::vector<std::shared_ptr<PatchInfo<BasicType>>> patches;
        std::unordered_map<const PatchInfo<BasicType> *, std::size_t> index;

        void insert(const std::shared_ptr<PatchInfo<BasicType>> &patch_info_ptr) {
            const PatchInfo<BasicType> *raw = patch_info_ptr.get();
            if (index.find(raw) != index.end()) {
                return;
            }
            index.emplace(raw, patches.size());
            patches.push_back(patch_info_ptr);
        }

        void erase(const std::shared_ptr<PatchInfo<BasicType>> &patch_info_ptr) {
            const PatchInfo<BasicType> *raw = patch_info_ptr.get();
            auto it = index.find(raw);
            if (it == index.end()) {
                return;
            }
            const std::size_t idx = it->second;
            const std::size_t last = patches.size() - 1;
            if (idx != last) {
                patches[idx] = patches[last];
                index[patches[idx].get()] = idx;
            }
            patches.pop_back();
            index.erase(it);
        }

        bool empty() const noexcept { return patches.empty(); }
    };

    void insert(std::shared_ptr<PatchInfo<BasicType>> &patch_info_ptr) {
        ++patch_key;
        patch_info_ptr->key = patch_key;
        {
            std::unique_lock<std::shared_mutex> ul_keyframe_set(keyframe_set_lock);
            all_keyframes_set.insert(patch_info_ptr->keyframe_ptr);
            if (patch_info_ptr->keyframe_ptr) {
                keyframe_by_frame_num_map[patch_info_ptr->keyframe_ptr->frame_num] = patch_info_ptr->keyframe_ptr;
            }
        }
        patch_info_ptr->keyframe_ptr->local_patches.insert(patch_info_ptr->key);
        patch_info_ptr->keyframe_ptr->associated_patches.insert(patch_info_ptr->key);
        {
            std::unique_lock<std::shared_mutex> ul_registry(patches_by_id_lock);
            patches_by_id[patch_info_ptr->key] = patch_info_ptr;
        }
        // insert the patch into hash map
        std::unique_lock<std::shared_mutex> ul_map(all_patches_map_lock);
        std::vector<GridKey> grid_indices = get_grid_indices(patch_info_ptr->get_enlarged_lower_bound_w(),
                                                             patch_info_ptr->get_enlarged_upper_bound_w());
        for (const auto &grid_idx : grid_indices) {
            // if (!contains(grid_idx)) {
            //     all_patches_map[grid_idx] =
            //     std::unordered_set<std::shared_ptr<PatchInfo<BasicType>>>{patch_info_ptr};
            // } else {
            all_patches_map[grid_idx].insert(patch_info_ptr);
            // }
        }
    }

    void insert(std::shared_ptr<PatchInfo<BasicType>> &patch_info_ptr, bool is_new_keyframe) {
        ++patch_key;
        patch_info_ptr->key = patch_key;
        if (is_new_keyframe) {
            std::unique_lock<std::shared_mutex> ul_keyframe_set(keyframe_set_lock);
            all_keyframes_set.insert(patch_info_ptr->keyframe_ptr);
            if (patch_info_ptr->keyframe_ptr) {
                keyframe_by_frame_num_map[patch_info_ptr->keyframe_ptr->frame_num] = patch_info_ptr->keyframe_ptr;
            }
        }
        patch_info_ptr->keyframe_ptr->local_patches.insert(patch_info_ptr->key);
        patch_info_ptr->keyframe_ptr->associated_patches.insert(patch_info_ptr->key);
        {
            std::unique_lock<std::shared_mutex> ul_registry(patches_by_id_lock);
            patches_by_id[patch_info_ptr->key] = patch_info_ptr;
        }
        // insert the patch into hash map
        std::unique_lock<std::shared_mutex> ul_map(all_patches_map_lock);
        std::vector<GridKey> grid_indices = get_grid_indices(patch_info_ptr->get_enlarged_lower_bound_w(),
                                                             patch_info_ptr->get_enlarged_upper_bound_w());
        for (const auto &grid_idx : grid_indices) {
            all_patches_map[grid_idx].insert(patch_info_ptr);
        }
    }

    void insert_patch(std::shared_ptr<PatchInfo<BasicType>> &patch_info_ptr) {
        ++patch_key;
        patch_info_ptr->key = patch_key;
        patch_info_ptr->keyframe_ptr->local_patches.insert(patch_info_ptr->key);
        patch_info_ptr->keyframe_ptr->associated_patches.insert(patch_info_ptr->key);
        {
            std::unique_lock<std::shared_mutex> ul_registry(patches_by_id_lock);
            patches_by_id[patch_info_ptr->key] = patch_info_ptr;
        }
        // insert the patch into hash map
        std::unique_lock<std::shared_mutex> ul_map(all_patches_map_lock);
        std::vector<GridKey> grid_indices = get_grid_indices(patch_info_ptr->get_enlarged_lower_bound_w(),
                                                             patch_info_ptr->get_enlarged_upper_bound_w());
        for (const auto &grid_idx : grid_indices) {
            all_patches_map[grid_idx].insert(patch_info_ptr);
        }
    }

    bool erase_patch(const std::shared_ptr<PatchInfo<BasicType>> &patch_info_ptr) {
        patch_info_ptr->get_enlarged_lower_bound_w();
        patch_info_ptr->get_enlarged_upper_bound_w();
        std::unique_lock<std::shared_mutex> ul_map(all_patches_map_lock);
        std::vector<GridKey> grid_indices = get_grid_indices(patch_info_ptr->get_enlarged_lower_bound_w(),
                                                             patch_info_ptr->get_enlarged_upper_bound_w());
        for (const auto &grid_idx : grid_indices) {
            auto target = all_patches_map.find(grid_idx);
            if (target != all_patches_map.end()) {
                target->second.erase(patch_info_ptr);
                if (target->second.empty()) { // if there are no patches in the grid, then remove the grid
                    all_patches_map.erase(target);
                }
            } else {
                std::cout << "Warning: the patch is not in the hash map!" << std::endl;
            }
        }
        patch_info_ptr->keyframe_ptr->local_patches.erase(patch_info_ptr->key);
        patch_info_ptr->keyframe_ptr->associated_patches.erase(patch_info_ptr->key);
        {
            std::unique_lock<std::shared_mutex> ul_registry(patches_by_id_lock);
            patches_by_id.erase(patch_info_ptr->key);
        }
        // even the keyframe is empty we need to keep it
        return patch_info_ptr->keyframe_ptr->local_patches.empty();
    }

    bool erase_patch_iterator(const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe_ptr,
                              typename std::unordered_set<PatchId>::iterator &patch_id_iter) {
        PatchId patch_id = *patch_id_iter;
        std::shared_ptr<PatchInfo<BasicType>> patch_info_ptr = get_patch_by_id(patch_id);
        if (!patch_info_ptr) {
            patch_id_iter = keyframe_ptr->local_patches.erase(patch_id_iter);
            keyframe_ptr->associated_patches.erase(patch_id);
            return keyframe_ptr->local_patches.empty();
        }
        if (patch_info_ptr->keyframe_ptr != keyframe_ptr) {
            patch_id_iter = keyframe_ptr->local_patches.erase(patch_id_iter);
            keyframe_ptr->associated_patches.erase(patch_id);
            return keyframe_ptr->local_patches.empty();
        }
        std::unique_lock<std::shared_mutex> ul_map(all_patches_map_lock);
        std::vector<GridKey> grid_indices = get_grid_indices(patch_info_ptr->get_enlarged_lower_bound_w(),
                                                             patch_info_ptr->get_enlarged_upper_bound_w());
        for (const auto &grid_idx : grid_indices) {
            auto target = all_patches_map.find(grid_idx);
            if (target != all_patches_map.end()) {
                target->second.erase(patch_info_ptr);
                if (target->second.empty()) { // if there are no patches in the grid, then remove the grid
                    all_patches_map.erase(target);
                }
            } else {
                std::cout << "Warning: the patch is not in the hash map!" << std::endl;
            }
        }
        patch_id_iter = keyframe_ptr->local_patches.erase(patch_id_iter);
        keyframe_ptr->associated_patches.erase(patch_info_ptr->key);
        {
            std::unique_lock<std::shared_mutex> ul_registry(patches_by_id_lock);
            patches_by_id.erase(patch_info_ptr->key);
        }
        // even the keyframe is empty we need to keep it
        return keyframe_ptr->local_patches.empty();
    }

    typename std::set<std::shared_ptr<KeyframeInfo<BasicType>>, key_frame_info_comparator<BasicType>>::iterator
    erase_keyframe(const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe_ptr) {
        std::unique_lock<std::shared_mutex> ul_keyframe_set(keyframe_set_lock);
        if (all_keyframes_set.find(keyframe_ptr) == all_keyframes_set.end()) {
            std::cerr << "Erased keyframe target doesn't exist!" << std::endl;
            return all_keyframes_set.begin();
        }
        for (auto iter = keyframe_ptr->local_patches.begin(); iter != keyframe_ptr->local_patches.end();) {
            if (auto patch_shared_ptr = get_patch_by_id(*iter)) {
                this->erase(patch_shared_ptr);
                iter = keyframe_ptr->local_patches.begin();
            } else {
                iter = keyframe_ptr->local_patches.erase(iter);
            }
        }
        all_keyframes_set.erase(keyframe_ptr);
        if (keyframe_ptr) {
            keyframe_by_frame_num_map.erase(keyframe_ptr->frame_num);
        }
        typename std::set<std::shared_ptr<KeyframeInfo<BasicType>>, key_frame_info_comparator<BasicType>>::iterator
            iter = all_keyframes_set.begin();
        return iter;
    }

    // if patch is using, then the keyframe will be kept
    void erase_old_patches(const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe_ptr) {
        std::shared_lock<std::shared_mutex> sl_keyframe_set(keyframe_set_lock);
        auto keyframe_ptr_iter = all_keyframes_set.find(keyframe_ptr);
        if (keyframe_ptr_iter == all_keyframes_set.end()) {
            std::cerr << "erase keyframe target doesn't exist!" << std::endl;
            return;
        }

        for (auto iter = keyframe_ptr->local_patches.begin(); iter != keyframe_ptr->local_patches.end();) {
            if (auto patch_shared_ptr = get_patch_by_id(*iter)) {
                std::vector<GridKey> grid_indices = get_grid_indices(patch_shared_ptr->get_enlarged_lower_bound_w(),
                                                                     patch_shared_ptr->get_enlarged_upper_bound_w());
                if (patch_shared_ptr.use_count() > (grid_indices.size() + 1)) {
                    ++iter;
                } else {
                    this->erase(patch_shared_ptr);
                    std::cout << "successfully delete a patch" << std::endl;
                    iter = keyframe_ptr->local_patches.begin();
                }
            } else {
                iter = keyframe_ptr->local_patches.erase(iter);
            }
        }
    }

    void erase_patch_caches_of_old_keyframes(int curent_fame_idx, int largest_frame_idx_difference) {
        std::shared_lock<std::shared_mutex> sl_keyframe_set(keyframe_set_lock);
        for (auto keyframe_ptr_iter = all_keyframes_set.begin();
             (curent_fame_idx - (*keyframe_ptr_iter)->frame_idx) >= largest_frame_idx_difference; ++keyframe_ptr_iter) {
            (*keyframe_ptr_iter)->clear_patches_content([this](PatchId id) { return get_patch_by_id(id); });
        }

        // for (auto keyframe_ptr_iter = all_keyframes_set.begin();
        //      (curent_fame_idx - (*keyframe_ptr_iter)->frame_idx) >= largest_frame_idx_difference;) {
        //     keyframe_ptr_iter = erase_keyframe(*keyframe_ptr_iter);
        //     std::cout << (*keyframe_ptr_iter)->frame_idx << std::endl;
        // }
    }

    void clear_patch_caches_out_sight(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &T_w_j,
                                      const std::pair<std::vector<double>, std::vector<double>> &valid_box_region) {
        std::shared_lock<std::shared_mutex> sl_keyframe_set(keyframe_set_lock);
        double valid_radius = std::sqrt(std::pow(valid_box_region.second[0] - valid_box_region.first[0], 2) +
                                        std::pow(valid_box_region.second[1] - valid_box_region.first[1], 2) +
                                        std::pow(valid_box_region.second[2] - valid_box_region.first[2], 2)) /
                              2.0;
        // NOTE: only keep the lastest 20 keyframes's raw point cloud
        int keyframe_total_size = all_keyframes_set.size();
        int current_frame_idx = (*all_keyframes_set.rbegin())->frame_idx;
        for (auto keyframe_ptr_iter = all_keyframes_set.begin(); keyframe_ptr_iter != all_keyframes_set.end();
             ++keyframe_ptr_iter) {
            (*keyframe_ptr_iter)->maintain_patches_content(
                T_w_j, valid_box_region, [this](PatchId id) { return get_patch_by_id(id); });
        }
    }

    void insert_keyframe(const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe_ptr) {
        std::unique_lock<std::shared_mutex> ul_keyframe_set(keyframe_set_lock);
        all_keyframes_set.insert(keyframe_ptr);
        if (keyframe_ptr) {
            keyframe_by_frame_num_map[keyframe_ptr->frame_num] = keyframe_ptr;
        }
    }

    bool is_contain_keyframe(const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe_ptr) {
        std::shared_lock<std::shared_mutex> sl_keyframe_set(keyframe_set_lock);
        return all_keyframes_set.find(keyframe_ptr) != all_keyframes_set.end();
    }

    auto begin() noexcept { return all_patches_map.begin(); }

    auto begin() const noexcept { return all_patches_map.begin(); }

    auto end() noexcept { return all_patches_map.end(); }

    auto end() const noexcept { return all_patches_map.end(); }

    auto back() noexcept { return all_patches_map.back(); }

    auto back() const noexcept { return all_patches_map.back(); }

    auto find(const GridKey &key) noexcept { return all_patches_map.find(key); }

    auto find(const GridKey &key) const noexcept { return all_patches_map.find(key); }

    auto find(const std::vector<int> &key) noexcept {
        if (key.size() < 3) {
            return all_patches_map.end();
        }
        return all_patches_map.find(GridKey{key[0], key[1], key[2]});
    }

    auto find(const std::vector<int> &key) const noexcept {
        if (key.size() < 3) {
            return all_patches_map.end();
        }
        return all_patches_map.find(GridKey{key[0], key[1], key[2]});
    }

    std::shared_ptr<PatchInfo<BasicType>> get_patch_by_id(PatchId id) const {
        std::shared_lock<std::shared_mutex> sl_registry(patches_by_id_lock);
        auto it = patches_by_id.find(id);
        if (it == patches_by_id.end()) {
            return nullptr;
        }
        return it->second;
    }

    std::vector<std::shared_ptr<KeyframeInfo<BasicType>>> keyframes_snapshot() const {
        std::shared_lock<std::shared_mutex> sl_keyframe_set(keyframe_set_lock);
        return {all_keyframes_set.begin(), all_keyframes_set.end()};
    }

    std::vector<std::shared_ptr<KeyframeInfo<BasicType>>> keyframes_snapshot_reverse() const {
        std::shared_lock<std::shared_mutex> sl_keyframe_set(keyframe_set_lock);
        std::vector<std::shared_ptr<KeyframeInfo<BasicType>>> result;
        result.reserve(all_keyframes_set.size());
        for (auto it = all_keyframes_set.rbegin(); it != all_keyframes_set.rend(); ++it) {
            result.push_back(*it);
        }
        return result;
    }

    std::shared_ptr<KeyframeInfo<BasicType>> latest_keyframe() const {
        std::shared_lock<std::shared_mutex> sl_keyframe_set(keyframe_set_lock);
        if (all_keyframes_set.empty()) {
            return nullptr;
        }
        return *all_keyframes_set.rbegin();
    }

    std::shared_ptr<KeyframeInfo<BasicType>> keyframe_by_frame_num(int frame_num) const {
        std::shared_lock<std::shared_mutex> sl_keyframe_set(keyframe_set_lock);
        if (frame_num < 0) {
            return nullptr;
        }
        auto it = keyframe_by_frame_num_map.find(frame_num);
        if (it != keyframe_by_frame_num_map.end()) {
            return it->second;
        }
        return nullptr;
    }

    PatchBucket &operator[](const GridKey &key) { return all_patches_map[key]; }

    const PatchBucket &operator[](const GridKey &key) const { return all_patches_map.at(key); }

    unsigned long size() const noexcept { return all_patches_map.size(); }
    unsigned long keyframe_size() const noexcept {
        std::shared_lock<std::shared_mutex> sl_keyframe_set(keyframe_set_lock);
        return all_keyframes_set.size();
    }
    unsigned long active_keyframe_size() const noexcept {
        std::shared_lock<std::shared_mutex> sl_keyframe_set(keyframe_set_lock);
        unsigned long counter = 0;
        for (auto iter = all_keyframes_set.rbegin(); iter != all_keyframes_set.rend(); ++iter) {
            if ((*iter)->is_active) {
                ++counter;
            } else {
                break;
            }
        }
        return counter;
    }

    void reserve(int n) { all_patches_map.reserve(n); }

    void clear() noexcept {
        std::unique_lock<std::shared_mutex> ul_keyframe_set(keyframe_set_lock);
        std::unique_lock<std::shared_mutex> ul_map(all_patches_map_lock);
        std::unique_lock<std::shared_mutex> ul_registry(patches_by_id_lock);
        all_patches_map.clear();
        patches_by_id.clear();
        all_keyframes_set.clear();
        keyframe_by_frame_num_map.clear();
    }

    bool empty() const noexcept { return all_patches_map.empty(); }

    bool contains(const GridKey &key) const { return all_patches_map.find(key) != all_patches_map.end(); }

    // aabb overlap
    std::vector<std::pair<std::shared_ptr<PatchInfo<BasicType>>, double>>
    query_data_association_overlap(aabb::AABB &bounding_box) {
        std::vector<std::pair<std::shared_ptr<PatchInfo<BasicType>>, double>> result;
        std::shared_lock<std::shared_mutex> sl_map(all_patches_map_lock);
        std::vector<GridKey> keys = get_grid_indices(bounding_box.lowerBound, bounding_box.upperBound);
        if (!keys.empty()) {
            for (auto &key : keys) {
                auto target = all_patches_map.find(key);
                if (target != all_patches_map.end()) {
                    // calculate the IoU of two intersected bounding box
                    for (const auto &patch_info_ptr : target->second.patches) {
                        double IoU_surface_area = curl::IoU_surface_area(
                            bounding_box.lowerBound, bounding_box.upperBound,
                            patch_info_ptr->get_enlarged_lower_bound_w(), patch_info_ptr->get_enlarged_upper_bound_w());
                        if (IoU_surface_area > 0) {
                            result.emplace_back(patch_info_ptr, IoU_surface_area);
                        }
                    }
                }
            }
        }

        return result;
    }

    // detect with enlarged box size
    std::vector<std::pair<std::shared_ptr<PatchInfo<BasicType>>, double>>
    query_data_association_overlap(std::vector<double> &lower_bound, std::vector<double> &upper_bound) {
        std::vector<std::pair<std::shared_ptr<PatchInfo<BasicType>>, double>> result;
        std::shared_lock<std::shared_mutex> sl_map(all_patches_map_lock);
        std::vector<GridKey> keys = get_grid_indices(lower_bound, upper_bound);
        if (!keys.empty()) {
            for (auto &key : keys) {
                auto target = all_patches_map.find(key);
                if (target != all_patches_map.end()) {
                    // calculate the IoU of two intersected bounding box
                    for (const auto &patch_info_ptr : target->second.patches) {
                        double IoU_surface_area = curl::IoU_surface_area(lower_bound, upper_bound,
                                                                         patch_info_ptr->get_enlarged_lower_bound_w(),
                                                                         patch_info_ptr->get_enlarged_upper_bound_w());
                        if (IoU_surface_area > 0) {
                            result.emplace_back(patch_info_ptr, IoU_surface_area);
                        }
                    }
                }
            }
        }

        if (!result.empty()) {
            // descending order, largest value be the first
            std::sort(
                result.begin(), result.end(),
                [](const std::pair<std::shared_ptr<PatchInfo<BasicType>>, double> &a,
                   const std::pair<std::shared_ptr<PatchInfo<BasicType>>, double> &b) { return a.second > b.second; });
        }
        return result;
    }

    // detect with enlarged box size
    void query_data_association_overlap_excludeCurrKeyframe(
        const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe_info_ptr, const std::vector<double> &lower_bound,
        const std::vector<double> &upper_bound,
        std::vector<std::pair<std::shared_ptr<PatchInfo<BasicType>>, double>> &result) {
        result.clear();
        std::shared_lock<std::shared_mutex> sl_map(all_patches_map_lock);
        std::vector<GridKey> keys = get_grid_indices(lower_bound, upper_bound);
        if (!keys.empty()) {
            for (auto &key : keys) {
                auto target = all_patches_map.find(key);
                if (target != all_patches_map.end()) {
                    // calculate the IoU of two intersected bounding box
                    for (const auto &patch_info_ptr : target->second.patches) {
                        if (keyframe_info_ptr->frame_idx == patch_info_ptr->keyframe_ptr->frame_idx) {
                            continue;
                        }
                        double IoU_surface_area = curl::IoU_surface_area(lower_bound, upper_bound,
                                                                         patch_info_ptr->get_enlarged_lower_bound_w(),
                                                                         patch_info_ptr->get_enlarged_upper_bound_w());
                        if (IoU_surface_area > 0) {
                            result.emplace_back(patch_info_ptr, IoU_surface_area);
                        }
                    }
                }
            }
        }

        if (!result.empty()) {
            // descending order, largest value be the first
            std::sort(
                result.begin(), result.end(),
                [](const std::pair<std::shared_ptr<PatchInfo<BasicType>>, double> &a,
                   const std::pair<std::shared_ptr<PatchInfo<BasicType>>, double> &b) { return a.second > b.second; });
        }
    }

    // detect with enlarged box size
    std::vector<std::pair<std::shared_ptr<PatchInfo<BasicType>>, double>>
    query_data_association_overlap_excludeCurrKeyframe(
        const std::shared_ptr<KeyframeInfo<BasicType>> &keyframe_info_ptr, const std::vector<double> &lower_bound,
        const std::vector<double> &upper_bound) {
        std::vector<std::pair<std::shared_ptr<PatchInfo<BasicType>>, double>> result;
        query_data_association_overlap_excludeCurrKeyframe(keyframe_info_ptr, lower_bound, upper_bound, result);
        return result;
    }

    // detect with original box size (original size used for plane data)
    std::vector<std::pair<std::shared_ptr<PatchInfo<BasicType>>, double>>
    query_data_association_true_overlap(std::vector<double> &lower_bound, std::vector<double> &upper_bound) {
        std::vector<std::pair<std::shared_ptr<PatchInfo<BasicType>>, double>> result;
        std::shared_lock<std::shared_mutex> sl_map(all_patches_map_lock);
        std::vector<GridKey> keys = get_grid_indices(lower_bound, upper_bound);
        if (!keys.empty()) {
            for (auto &key : keys) {
                auto target = all_patches_map.find(key);
                if (target != all_patches_map.end()) {
                    // calculate the IoU of two intersected bounding box
                    for (const auto &patch_info_ptr : target->second.patches) {
                        double IoU_surface_area =
                            curl::IoU_surface_area(lower_bound, upper_bound, patch_info_ptr->get_lower_bound_w(),
                                                   patch_info_ptr->get_upper_bound_w());
                        if (IoU_surface_area > 0) {
                            result.emplace_back(patch_info_ptr, IoU_surface_area);
                        }
                    }
                }
            }
        }

        if (!result.empty()) {
            // descending order, largest value be the first
            std::sort(
                result.begin(), result.end(),
                [](const std::pair<std::shared_ptr<PatchInfo<BasicType>>, double> &a,
                   const std::pair<std::shared_ptr<PatchInfo<BasicType>>, double> &b) { return a.second > b.second; });
        }
        return result;
    }

    std::vector<int> get_grid_indices(const std::vector<double> &point) {
        std::vector<int> grid_idx(3);
        for (int i = 0; i < 3; ++i) {
            grid_idx[i] = std::ceil(point[i] / voxel_length);
        }
        // go through grids inside the bounding box
        return grid_idx;
    }

    std::unordered_map<std::vector<int>, std::vector<Eigen::Vector3<BasicType>>, VoxelHashFuncPrime>
    get_grid_indices(const Eigen::MatrixX<BasicType> &points) {
        std::unordered_map<std::vector<int>, std::vector<Eigen::Vector3<BasicType>>, VoxelHashFuncPrime> grid_points;
        std::vector<int> grid_idx(3);
        for (int i = 0; i < points.cols(); i++) {
            grid_idx[0] = std::ceil(points(0, i) / voxel_length);
            grid_idx[1] = std::ceil(points(1, i) / voxel_length);
            grid_idx[2] = std::ceil(points(2, i) / voxel_length);
            grid_points[grid_idx].emplace_back(points(0, i), points(1, i), points(2, i));
        }
        return grid_points;
    }

    void save_keyframe_trajectory(const std::string &filename) {
        std::shared_lock<std::shared_mutex> sl_keyframe_set(keyframe_set_lock);
        std::string kitti_filename = filename + "/keyframe_trajectory.txt";
        std::ofstream kitti_outfile(kitti_filename);
        for (const auto &keyframe_iter : all_keyframes_set) {
            const auto &pose_matrix = keyframe_iter->get_T_w_lidar();
            kitti_outfile << keyframe_iter->frame_idx << " ";
            for (int i = 0; i < 3; ++i) { // We only need the first 3 rows
                for (int j = 0; j < 4; ++j) {
                    kitti_outfile << pose_matrix(i, j);
                    if (!(i == 2 && j == 3)) { // Avoid trailing space at the end
                        kitti_outfile << " ";
                    }
                }
            }
            kitti_outfile << "\n";
        }
        kitti_outfile.close();
    }

    void save_optimized_trajectories(const std::string &filename) {
        std::shared_lock<std::shared_mutex> sl_keyframe_set(keyframe_set_lock);
        std::string kitti_filename = filename + "/optimized_trajectory.txt";
        std::ofstream kitti_outfile(kitti_filename);
        for (const auto &keyframe_iter : all_keyframes_set) {
            const auto pose_matrix = keyframe_iter->get_graph_pose_w_lidar_T().matrix();

            kitti_outfile << keyframe_iter->frame_idx << " ";
            for (int i = 0; i < 3; ++i) { // We only need the first 3 rows
                for (int j = 0; j < 4; ++j) {
                    kitti_outfile << pose_matrix(i, j);
                    if (!(i == 2 && j == 3)) { // Avoid trailing space at the end
                        kitti_outfile << " ";
                    }
                }
            }
            kitti_outfile << "\n";
        }
        kitti_outfile.close();
    }

    void save_g2o_trajectories(const std::string &filename) {
        std::shared_lock<std::shared_mutex> sl_keyframe_set(keyframe_set_lock);
        std::ofstream g2o_file(filename);
        for (const auto &keyframe_iter : all_keyframes_set) {
            g2o_file << "VERTEX_SE3:QUAT " << keyframe_iter->frame_idx << " "
                     << keyframe_iter->get_graph_pose_w_lidar().p.x() << " "
                     << keyframe_iter->get_graph_pose_w_lidar().p.y() << " "
                     << keyframe_iter->get_graph_pose_w_lidar().p.z() << " "
                     << keyframe_iter->get_graph_pose_w_lidar().q.x() << " "
                     << keyframe_iter->get_graph_pose_w_lidar().q.y() << " "
                     << keyframe_iter->get_graph_pose_w_lidar().q.z() << " "
                     << keyframe_iter->get_graph_pose_w_lidar().q.w() << "\n";
        }
        for (auto keyframe_iter = std::next(all_keyframes_set.begin(), 1); keyframe_iter != all_keyframes_set.end();
             ++keyframe_iter) {
            const auto last_keyframe_iter = std::next(keyframe_iter, -1);
            g2o_file << "EDGE_SE3:QUAT " << (*last_keyframe_iter)->frame_idx << " " << (*keyframe_iter)->frame_idx
                     << " ";
            g2o_file << (*keyframe_iter)->get_graph_obs_j_1_j().p.x() << " "
                     << (*keyframe_iter)->get_graph_obs_j_1_j().p.y() << " "
                     << (*keyframe_iter)->get_graph_obs_j_1_j().p.z() << " "
                     << (*keyframe_iter)->get_graph_obs_j_1_j().q.x() << " "
                     << (*keyframe_iter)->get_graph_obs_j_1_j().q.y() << " "
                     << (*keyframe_iter)->get_graph_obs_j_1_j().q.z() << " "
                     << (*keyframe_iter)->get_graph_obs_j_1_j().q.w() << " ";
            for (int i = 6; i > 0; --i) {
                g2o_file << 1 << " ";
                for (int j = 0; j < i - 1; ++j) {
                    g2o_file << 0 << " ";
                }
            }
            g2o_file << "\n";
        }
        g2o_file.close();
    }

    void update_keyframe_pose_by_poseGraph() {
        std::unique_lock<std::shared_mutex> ul_T_w_lidar(KeyframeInfo<BasicType>::T_w_lidar_lock);
        // read lock for the keyframe set
        std::shared_lock<std::shared_mutex> sl_keyframe_set(keyframe_set_lock);
        std::unique_lock<std::shared_mutex> ul_map(all_patches_map_lock);
        struct PatchGridUpdate {
            std::shared_ptr<PatchInfo<BasicType>> patch_ptr;
            GridBounds old_bounds;
            GridBounds new_bounds;
        };
        std::vector<PatchGridUpdate> grid_updates;
        grid_updates.reserve(1024);
        for (auto keyframe_iter = all_keyframes_set.begin(); keyframe_iter != all_keyframes_set.end(); ++keyframe_iter) {
            Eigen::Matrix4d T_w_lidar_old = (*keyframe_iter)->get_T_w_lidar_noLock();
            (*keyframe_iter)->update_T_w_lidar_with_graph_pose_noLock();
            Eigen::Matrix4d T_w_lidar_new = (*keyframe_iter)->get_T_w_lidar_noLock();
            Eigen::Matrix4d T_wNew_wOld = T_w_lidar_new * Eigen::Isometry3d(T_w_lidar_old).inverse().matrix();
            for (const auto &patch_id : (*keyframe_iter)->local_patches) {
                if (auto patch_ptr = get_patch_by_id(patch_id)) {
                    std::vector<double> old_enlarged_lower = patch_ptr->get_enlarged_lower_bound_w();
                    std::vector<double> old_enlarged_upper = patch_ptr->get_enlarged_upper_bound_w();
                    GridBounds old_bounds = get_grid_bounds(old_enlarged_lower, old_enlarged_upper);
                    auto new_enlarged_bounds =
                        transform_aabb_bounds(old_enlarged_lower, old_enlarged_upper, T_wNew_wOld);
                    GridBounds new_bounds = get_grid_bounds(new_enlarged_bounds.first, new_enlarged_bounds.second);

                    patch_ptr->set_enlarged_lower_bound_w(new_enlarged_bounds.first);
                    patch_ptr->set_enlarged_upper_bound_w(new_enlarged_bounds.second);
                    std::vector<double> old_lower = patch_ptr->get_lower_bound_w();
                    std::vector<double> old_upper = patch_ptr->get_upper_bound_w();
                    auto new_bounds_raw = transform_aabb_bounds(old_lower, old_upper, T_wNew_wOld);
                    patch_ptr->set_lower_bound_w(new_bounds_raw.first);
                    patch_ptr->set_upper_bound_w(new_bounds_raw.second);

                    if (old_bounds != new_bounds) {
                        grid_updates.push_back(PatchGridUpdate{patch_ptr, old_bounds, new_bounds});
                    }
                    // update initial points before fixing the projection plane
                    if (!patch_ptr->initial_points_w_vec.empty()) {
                        for (auto &points_w : patch_ptr->initial_points_w_vec) {
                            points_w = ((T_wNew_wOld(Eigen::seq(0, 2), Eigen::seq(0, 2)).cast<BasicType>() * points_w)
                                            .colwise() +
                                        T_wNew_wOld(Eigen::seq(0, 2), 3).cast<BasicType>())
                                           .eval();
                        }
                    }
                }
            }
        }
        if (!grid_updates.empty()) {
            for (const auto &update : grid_updates) {
                const std::vector<GridKey> grid_indices_old = get_grid_indices(update.old_bounds);
                for (const auto &grid_idx : grid_indices_old) {
                    auto target = all_patches_map.find(grid_idx);
                    if (target != all_patches_map.end()) {
                        target->second.erase(update.patch_ptr);
                        if (target->second.empty()) {
                            all_patches_map.erase(target);
                        }
                    }
                }
                const std::vector<GridKey> grid_indices_new = get_grid_indices(update.new_bounds);
                for (const auto &grid_idx : grid_indices_new) {
                    all_patches_map[grid_idx].insert(update.patch_ptr);
                }
            }
        }
        std::cout << "update keyframe pose and bounding box by pose graph" << std::endl;
    }

    void apply_delta_to_non_backend_keyframes(const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &delta) {
        std::unique_lock<std::shared_mutex> ul_T_w_lidar(KeyframeInfo<BasicType>::T_w_lidar_lock);
        std::shared_lock<std::shared_mutex> sl_keyframe_set(keyframe_set_lock);
        std::unique_lock<std::shared_mutex> ul_map(all_patches_map_lock);
        struct PatchGridUpdate {
            std::shared_ptr<PatchInfo<BasicType>> patch_ptr;
            GridBounds old_bounds;
            GridBounds new_bounds;
        };
        std::vector<PatchGridUpdate> grid_updates;
        grid_updates.reserve(1024);
        for (auto keyframe_iter = all_keyframes_set.rbegin(); keyframe_iter != all_keyframes_set.rend();
             ++keyframe_iter) {
            if (!(*keyframe_iter)) {
                continue;
            }
            if ((*keyframe_iter)->is_backend_keyframe) {
                break;
            }
            Eigen::Matrix4d T_w_lidar_old = (*keyframe_iter)->get_T_w_lidar_noLock();
            Eigen::Matrix4d T_w_lidar_new = delta * T_w_lidar_old;
            (*keyframe_iter)->set_T_w_lidar_noLock(T_w_lidar_new);
            Eigen::Matrix4d T_wNew_wOld = T_w_lidar_new * Eigen::Isometry3d(T_w_lidar_old).inverse().matrix();
            for (const auto &patch_id : (*keyframe_iter)->local_patches) {
                if (auto patch_ptr = get_patch_by_id(patch_id)) {
                    std::vector<double> old_enlarged_lower = patch_ptr->get_enlarged_lower_bound_w();
                    std::vector<double> old_enlarged_upper = patch_ptr->get_enlarged_upper_bound_w();
                    GridBounds old_bounds = get_grid_bounds(old_enlarged_lower, old_enlarged_upper);
                    auto new_enlarged_bounds =
                        transform_aabb_bounds(old_enlarged_lower, old_enlarged_upper, T_wNew_wOld);
                    GridBounds new_bounds = get_grid_bounds(new_enlarged_bounds.first, new_enlarged_bounds.second);

                    patch_ptr->set_enlarged_lower_bound_w(new_enlarged_bounds.first);
                    patch_ptr->set_enlarged_upper_bound_w(new_enlarged_bounds.second);
                    std::vector<double> old_lower = patch_ptr->get_lower_bound_w();
                    std::vector<double> old_upper = patch_ptr->get_upper_bound_w();
                    auto new_bounds_raw = transform_aabb_bounds(old_lower, old_upper, T_wNew_wOld);
                    patch_ptr->set_lower_bound_w(new_bounds_raw.first);
                    patch_ptr->set_upper_bound_w(new_bounds_raw.second);

                    if (old_bounds != new_bounds) {
                        grid_updates.push_back(PatchGridUpdate{patch_ptr, old_bounds, new_bounds});
                    }
                    if (!patch_ptr->initial_points_w_vec.empty()) {
                        for (auto &points_w : patch_ptr->initial_points_w_vec) {
                            points_w = ((T_wNew_wOld(Eigen::seq(0, 2), Eigen::seq(0, 2)).cast<BasicType>() * points_w)
                                            .colwise() +
                                        T_wNew_wOld(Eigen::seq(0, 2), 3).cast<BasicType>())
                                           .eval();
                        }
                    }
                }
            }
        }
        if (!grid_updates.empty()) {
            for (const auto &update : grid_updates) {
                const std::vector<GridKey> grid_indices_old = get_grid_indices(update.old_bounds);
                for (const auto &grid_idx : grid_indices_old) {
                    auto target = all_patches_map.find(grid_idx);
                    if (target != all_patches_map.end()) {
                        target->second.erase(update.patch_ptr);
                        if (target->second.empty()) {
                            all_patches_map.erase(target);
                        }
                    }
                }
                const std::vector<GridKey> grid_indices_new = get_grid_indices(update.new_bounds);
                for (const auto &grid_idx : grid_indices_new) {
                    all_patches_map[grid_idx].insert(update.patch_ptr);
                }
            }
        }
    }

    void update_patch_keyframe(const std::shared_ptr<PatchInfo<BasicType>> &patch_info_ptr) {
        std::shared_lock<std::shared_mutex> sl_keyframe_set(keyframe_set_lock);
        int frame_num = std::round((static_cast<double>(patch_info_ptr->create_keyframe_num) +
                                    static_cast<double>(patch_info_ptr->last_update_keyframe_num)) /
                                   2.0);
        if (frame_num != patch_info_ptr->keyframe_ptr->frame_num) {
            if (frame_num < 0 || static_cast<std::size_t>(frame_num) >= all_keyframes_set.size()) {
                return;
            }
            auto iter = all_keyframes_set.begin();
            std::advance(iter, frame_num);
            std::shared_ptr<KeyframeInfo<BasicType>> target_keyframe_ptr = *iter;
            assert(frame_num == target_keyframe_ptr->frame_num);
            if (is_patch_rebind_frozen_for_swap(patch_info_ptr, target_keyframe_ptr)) {
                return;
            }
            swap_patch(patch_info_ptr, target_keyframe_ptr);
        }
    }

    void begin_skip_patch_rebind_for_keyframes(const std::unordered_set<int> &frame_nums) {
        if (frame_nums.empty()) {
            return;
        }
        std::lock_guard<std::mutex> lock(patch_rebind_freeze_lock);
        for (const int frame_num : frame_nums) {
            ++patch_rebind_freeze_refcount[frame_num];
        }
    }

    void end_skip_patch_rebind_for_keyframes(const std::unordered_set<int> &frame_nums) {
        if (frame_nums.empty()) {
            return;
        }
        std::lock_guard<std::mutex> lock(patch_rebind_freeze_lock);
        for (const int frame_num : frame_nums) {
            auto it = patch_rebind_freeze_refcount.find(frame_num);
            if (it == patch_rebind_freeze_refcount.end()) {
                continue;
            }
            if (it->second <= 1) {
                patch_rebind_freeze_refcount.erase(it);
            } else {
                --(it->second);
            }
        }
    }

    void force_swap_patch(const std::shared_ptr<PatchInfo<BasicType>> &patch_info_ptr,
                          const std::shared_ptr<KeyframeInfo<BasicType>> &target_keyframe_ptr) {
        swap_patch(patch_info_ptr, target_keyframe_ptr);
    }

    bool loop_closure_valid_check(int shift_idx, double squared_distance) {
        std::shared_lock<std::shared_mutex> sl_keyframe_set(keyframe_set_lock);
        if (all_keyframes_set.empty()) {
            return false;
        }
        const int size = static_cast<int>(all_keyframes_set.size());
        const int detect_idx = static_cast<int>(std::round((shift_idx + size - 1) / 2.0));
        if (detect_idx < 0 || detect_idx >= size) {
            return false;
        }
        auto detect_keyframe_iter = all_keyframes_set.begin();
        std::advance(detect_keyframe_iter, detect_idx);
        auto latest_keyframe_ptr = *all_keyframes_set.rbegin();
        if (((*detect_keyframe_iter)->get_T_w_lidar()(Eigen::seq(0, 2), 3) -
             latest_keyframe_ptr->get_T_w_lidar()(Eigen::seq(0, 2), 3))
                .squaredNorm() < squared_distance) { // the middle keyframe needs to be far
            return false;
        } else {
            return true;
        }
    }

    bool loop_closure_valid_check(int shift_idx, int query_idx, double squared_distance) {
        std::shared_lock<std::shared_mutex> sl_keyframe_set(keyframe_set_lock);
        if (query_idx < 0 || query_idx >= static_cast<int>(all_keyframes_set.size())) {
            return false;
        }
        int detect_idx = std::round((shift_idx + query_idx) / 2.0);
        if (detect_idx < 0 || detect_idx >= static_cast<int>(all_keyframes_set.size())) {
            return false;
        }
        auto detect_keyframe_iter = all_keyframes_set.begin();
        std::advance(detect_keyframe_iter, detect_idx);
        auto query_keyframe_iter = all_keyframes_set.begin();
        std::advance(query_keyframe_iter, query_idx);
        if (((*detect_keyframe_iter)->get_T_w_lidar()(Eigen::seq(0, 2), 3) -
             (*query_keyframe_iter)->get_T_w_lidar()(Eigen::seq(0, 2), 3))
                .squaredNorm() < squared_distance) {
            return false;
        } else {
            return true;
        }
    }

    void clear_history_keyframe_clouds() {
        std::shared_lock<std::shared_mutex> sl_keyframe_set(keyframe_set_lock);
        if (!all_keyframes_set.empty()) {
            auto keyframe_iter = all_keyframes_set.rbegin();
            unsigned int current_label = (*keyframe_iter)->trajectory_label_ptr->label_frame_num;
            for (; keyframe_iter != all_keyframes_set.rend(); ++keyframe_iter) {
                if ((*keyframe_iter)->trajectory_label_ptr->label_frame_num != current_label) {
                    if ((*keyframe_iter)->seg_cloud_ptr == nullptr && (*keyframe_iter)->ground_cloud_ptr == nullptr) {
                        break;
                    }
                    // Skip keyframes being used by BA
                    if ((*keyframe_iter)->is_being_used_by_ba.load()) {
                        continue;
                    }
                    (*keyframe_iter)->seg_cloud_ptr.reset();
                    (*keyframe_iter)->ground_cloud_ptr.reset();
                }
            }
        }
    }

    mutable std::shared_mutex keyframe_set_lock;

    // TODO: implement the update of the spatial hashing grid after loop closure

  private:
    GridBounds get_grid_bounds(const std::vector<double> &lower_bound, const std::vector<double> &upper_bound) const {
        assert((lower_bound.size() == 3) && (upper_bound.size() == 3));
        GridBounds bounds;
        bounds.x_min = std::ceil(lower_bound[0] / voxel_length);
        bounds.y_min = std::ceil(lower_bound[1] / voxel_length);
        bounds.z_min = std::ceil(lower_bound[2] / voxel_length);
        bounds.x_max = std::ceil(upper_bound[0] / voxel_length);
        bounds.y_max = std::ceil(upper_bound[1] / voxel_length);
        bounds.z_max = std::ceil(upper_bound[2] / voxel_length);
        return bounds;
    }

    std::vector<GridKey> get_grid_indices(const GridBounds &bounds) const {
        const int x_len = bounds.x_max - bounds.x_min + 1;
        const int y_len = bounds.y_max - bounds.y_min + 1;
        const int z_len = bounds.z_max - bounds.z_min + 1;
        std::vector<GridKey> grid_indices;
        if (x_len <= 0 || y_len <= 0 || z_len <= 0) {
            return grid_indices;
        }
        grid_indices.reserve(static_cast<std::size_t>(x_len * y_len * z_len));
        for (int x = bounds.x_min; x <= bounds.x_max; ++x) {
            for (int y = bounds.y_min; y <= bounds.y_max; ++y) {
                for (int z = bounds.z_min; z <= bounds.z_max; ++z) {
                    grid_indices.push_back(GridKey{x, y, z});
                }
            }
        }
        return grid_indices;
    }

    std::vector<GridKey> get_grid_indices(const std::vector<double> &lower_bound,
                                          const std::vector<double> &upper_bound) {
        return get_grid_indices(get_grid_bounds(lower_bound, upper_bound));
    }

    std::pair<std::vector<double>, std::vector<double>>
    transform_aabb_bounds(const std::vector<double> &lower_bound, const std::vector<double> &upper_bound,
                          const Eigen::Matrix4d &T_target_source) const {
        assert((lower_bound.size() == 3) && (upper_bound.size() == 3));
        Eigen::Vector4d corner_h(0.0, 0.0, 0.0, 1.0);
        Eigen::Vector3d transformed_lower;
        Eigen::Vector3d transformed_upper;
        bool is_first_corner = true;
        for (int xi = 0; xi < 2; ++xi) {
            corner_h.x() = xi == 0 ? lower_bound[0] : upper_bound[0];
            for (int yi = 0; yi < 2; ++yi) {
                corner_h.y() = yi == 0 ? lower_bound[1] : upper_bound[1];
                for (int zi = 0; zi < 2; ++zi) {
                    corner_h.z() = zi == 0 ? lower_bound[2] : upper_bound[2];
                    const Eigen::Vector3d transformed_corner = (T_target_source * corner_h).head<3>();
                    if (is_first_corner) {
                        transformed_lower = transformed_corner;
                        transformed_upper = transformed_corner;
                        is_first_corner = false;
                    } else {
                        transformed_lower = transformed_lower.cwiseMin(transformed_corner);
                        transformed_upper = transformed_upper.cwiseMax(transformed_corner);
                    }
                }
            }
        }
        std::vector<double> lower_bound_out(3), upper_bound_out(3);
        Eigen::Map<Eigen::Vector3d>(lower_bound_out.data()) = transformed_lower;
        Eigen::Map<Eigen::Vector3d>(upper_bound_out.data()) = transformed_upper;
        return std::make_pair(lower_bound_out, upper_bound_out);
    }

    void update_patch_bounding_box(const std::shared_ptr<PatchInfo<BasicType>> &patch_info_ptr,
                                   const Eigen::Vector3d &displacement) {
        std::vector<double> old_enlarged_lower = patch_info_ptr->get_enlarged_lower_bound_w();
        std::vector<double> old_enlarged_upper = patch_info_ptr->get_enlarged_upper_bound_w();
        GridBounds old_bounds = get_grid_bounds(old_enlarged_lower, old_enlarged_upper);
        std::vector<double> enlarged_lower_bound_w_new(3), enlarged_upper_bound_w_new(3);
        Eigen::Map<Eigen::Vector3d>(enlarged_lower_bound_w_new.data()) =
            Eigen::Map<Eigen::Vector3d>(old_enlarged_lower.data()) + displacement;
        Eigen::Map<Eigen::Vector3d>(enlarged_upper_bound_w_new.data()) =
            Eigen::Map<Eigen::Vector3d>(old_enlarged_upper.data()) + displacement;
        GridBounds new_bounds = get_grid_bounds(enlarged_lower_bound_w_new, enlarged_upper_bound_w_new);

        if (old_bounds != new_bounds) {
            std::unique_lock<std::shared_mutex> ul_map(all_patches_map_lock);
            // remove the old patch from spatial hashing map
            std::vector<GridKey> grid_indices_old = get_grid_indices(old_bounds);
            for (const auto &grid_idx : grid_indices_old) {
                auto target = all_patches_map.find(grid_idx);
                if (target != all_patches_map.end()) {
                    target->second.erase(patch_info_ptr);
                    if (target->second.empty()) { // if there are no patches in the grid, then remove the grid
                        all_patches_map.erase(target);
                    }
                }
            }
            // insert the new patch into spatial hashing map
            std::vector<GridKey> grid_indices_new = get_grid_indices(new_bounds);
            for (const auto &grid_idx : grid_indices_new) {
                all_patches_map[grid_idx].insert(patch_info_ptr);
            }
        }
        // update PatchInfo
        patch_info_ptr->set_enlarged_lower_bound_w(enlarged_lower_bound_w_new);
        patch_info_ptr->set_enlarged_upper_bound_w(enlarged_upper_bound_w_new);
        std::vector<double> lower_bound_w_new(3), upper_bound_w_new(3);
        Eigen::Map<Eigen::Vector3d>(lower_bound_w_new.data()) =
            Eigen::Map<Eigen::Vector3d>((patch_info_ptr->get_lower_bound_w()).data()) + displacement;
        Eigen::Map<Eigen::Vector3d>(upper_bound_w_new.data()) =
            Eigen::Map<Eigen::Vector3d>((patch_info_ptr->get_upper_bound_w()).data()) + displacement;
        patch_info_ptr->set_lower_bound_w(lower_bound_w_new);
        patch_info_ptr->set_upper_bound_w(upper_bound_w_new);
    }

    void swap_patch(const std::shared_ptr<PatchInfo<BasicType>> &patch_info_ptr,
                    const std::shared_ptr<KeyframeInfo<BasicType>> &target_keyframe_ptr) {
        std::unique_lock<std::mutex> ul_local_patches(patch_info_ptr->keyframe_ptr->local_patches_lock);
        std::unique_lock<std::mutex> ul_target_local_patches(target_keyframe_ptr->local_patches_lock);
        // calculate the new T_obj_lidar
        Eigen::Isometry3d T_obj_w =
            patch_info_ptr->T_obj_lidar * Eigen::Isometry3d(patch_info_ptr->keyframe_ptr->get_T_w_lidar()).inverse();
        Eigen::Isometry3d T_obj_lidar_new = T_obj_w * Eigen::Isometry3d(target_keyframe_ptr->get_T_w_lidar());
        // remove the patch from the original keyframe
        PatchId patch_id = patch_info_ptr->key;
        patch_info_ptr->keyframe_ptr->local_patches.erase(patch_id);
        // insert patch into the target keyframe
        target_keyframe_ptr->local_patches.insert(patch_id);
        patch_info_ptr->keyframe_ptr = target_keyframe_ptr;
        patch_info_ptr->T_obj_lidar = T_obj_lidar_new;
    }

    bool is_patch_rebind_frozen_unlocked(int frame_num) const {
        return patch_rebind_freeze_refcount.find(frame_num) != patch_rebind_freeze_refcount.end();
    }

    bool is_patch_rebind_frozen_for_swap(const std::shared_ptr<PatchInfo<BasicType>> &patch_info_ptr,
                                         const std::shared_ptr<KeyframeInfo<BasicType>> &target_keyframe_ptr) const {
        if (!patch_info_ptr || !patch_info_ptr->keyframe_ptr || !target_keyframe_ptr) {
            return false;
        }
        std::lock_guard<std::mutex> lock(patch_rebind_freeze_lock);
        return is_patch_rebind_frozen_unlocked(patch_info_ptr->keyframe_ptr->frame_num) ||
               is_patch_rebind_frozen_unlocked(target_keyframe_ptr->frame_num);
    }

    std::unordered_map<GridKey, PatchBucket, GridKeyHash, GridKeyEqual> all_patches_map;
    mutable std::shared_mutex all_patches_map_lock;
    std::unordered_map<PatchId, std::shared_ptr<PatchInfo<BasicType>>> patches_by_id;
    mutable std::shared_mutex patches_by_id_lock;
    mutable std::mutex patch_rebind_freeze_lock;
    std::unordered_map<int, std::size_t> patch_rebind_freeze_refcount;
    // the comparator make the keyframes has the order of time

  public:
    std::set<std::shared_ptr<KeyframeInfo<BasicType>>, key_frame_info_comparator<BasicType>> all_keyframes_set;
    std::unordered_map<int, std::shared_ptr<KeyframeInfo<BasicType>>> keyframe_by_frame_num_map;
    // voxel size
    double voxel_length;

    PatchId patch_key;

    // template <class Archive> void serialize(Archive &ar, const unsigned int version) { ar &all_saving_patches_ptr; }

    void all_saving_patches() {
        std::shared_lock<std::shared_mutex> sl_keyframe_set(keyframe_set_lock);
        for (const auto &keyframe_ptr : all_keyframes_set) {
            for (const auto &patch_id : keyframe_ptr->local_patches) {
                if (auto patch_shared_ptr = get_patch_by_id(patch_id)) {
                    all_saving_patches_ptr.insert(patch_shared_ptr);
                }
            }
        }
    }

    std::unordered_set<std::shared_ptr<PatchInfo<BasicType>>> all_saving_patches_ptr;

  private:
    // friend class boost::serialization::access;
};

// Export the instantiated template class
// BOOST_CLASS_EXPORT(SpatialHashing<BT>)

#endif // SPATIAL_HASHING_H
