#pragma once

#include <assert.h>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

namespace cir {

template <typename Node>
struct NodeIndex {
    uint32_t valid : 1;
    uint32_t idx : 31;

    NodeIndex() : valid(0), idx(0) {}
    explicit NodeIndex(uint32_t idx) : valid(1), idx(idx) {}

    inline static NodeIndex null() {
        return {};
    }

    inline bool isValid() const {
        return valid == 1;
    }

    // Needed for std::set
    bool operator<(const NodeIndex& other) const {
        return std::tie(valid, idx) < std::tie(other.valid, other.idx);
    }
};

template <typename Node>
struct NodeVector {
public:
    Node& get(NodeIndex<Node> index) {
        assert(index.valid == 1);
        return m_storage[index.idx];
    }

    const Node& get(NodeIndex<Node> index) const {
        assert(index.valid == 1);
        return m_storage[index.idx];
    }

    template <typename... Args>
    NodeIndex<Node> emplace(Args&&...args) {
        NodeIndex<Node> new_index(m_storage.size());
        m_storage.emplace_back(std::forward<Args>(args)...);
        return new_index;
    }

private:
    std::vector<Node> m_storage;
};

template <typename... Nodes>
struct GenericAst {
    std::tuple<NodeVector<Nodes>...> nodes;

    GenericAst() : nodes() {}

    template <typename Node, typename... Args>
    NodeIndex<Node> emplaceNode(Args&&...args) {
        auto& node_vector = getNodeVector<Node>();
        auto index = node_vector.emplace(std::forward<Args>(args)...);
        return NodeIndex(index);
    }

    template <typename Node>
    Node& getNode(NodeIndex<Node> index) {
        auto& node_vector = getNodeVector<Node>();
        return node_vector.get(index);
    }

    template <typename Node>
    const Node& getNode(NodeIndex<Node> index) const {
        const auto& node_vector = getNodeVector<Node>();
        return node_vector.get(index);
    }

protected:
    template <typename Node>
    NodeVector<Node>& getNodeVector() {
        return std::get<NodeVector<Node>>(nodes);
    }

    template <typename Node>
    const NodeVector<Node>& getNodeVector() const {
        return std::get<NodeVector<Node>>(nodes);
    }
};

} // namespace cir
