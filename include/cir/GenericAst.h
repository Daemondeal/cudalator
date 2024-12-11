#pragma once

#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

namespace cir {

using NodeIndex = uint32_t;

template <typename Node>
struct NodeVector {
public:
    Node& get(NodeIndex index) {
        return m_storage[index];
    }

    const Node& get(NodeIndex index) const {
        return m_storage[index];
    }

    NodeIndex *getPtr(NodeIndex index) {
        return &m_storage[index];
    }

    NodeIndex add(const Node& node) {
        NodeIndex new_index = static_cast<NodeIndex>(m_storage.size());
        m_storage.push_back(node);

        return new_index;
    }

    template <typename... Args>
    NodeIndex emplace(Args&&...args) {
        NodeIndex new_index = static_cast<NodeIndex>(m_storage.size());
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

    template <typename Node>
    NodeIndex addNode(const Node& node) {
        auto& node_vector = getNodeVector<Node>();
        return node_vector.add(node);
    }

    template <typename Node, typename... Args>
    NodeIndex *emplaceNode(Args&&...args) {
        auto& node_vector = getNodeVector<Node>();
        auto index = node_vector.emplace(std::forward<Args>(args)...);
        return node_vector.getPtr(index);
    }

    template <typename Node>
    Node& getNode(NodeIndex index) {
        auto& node_vector = getNodeVector<Node>();
        return node_vector.get(index);
    }

    template <typename Node>
    const Node& getNode(NodeIndex index) const {
        const auto& node_vector = getNodeVector<Node>();
        return node_vector.get(index);
    }



private:
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
