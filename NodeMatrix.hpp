#include <vector>

class NodeMatrix {
public:
    using NodeId = unsigned;

    explicit NodeMatrix(const ExactCoverProblem& problem);

    void cover_column(NodeId id);
    void uncover_column(NodeId id);

    auto width() const -> unsigned;
    auto root_id() const -> NodeId;

    auto X(NodeId id) const -> unsigned;
    auto Y(NodeId id) const -> unsigned;
    auto S(NodeId id) const -> unsigned;
    auto C(NodeId id) const -> NodeId;
    auto L(NodeId id) const -> NodeId;
    auto R(NodeId id) const -> NodeId;
    auto U(NodeId id) const -> NodeId;
    auto D(NodeId id) const -> NodeId;

private:
    void add_row(unsigned y, const std::vector<unsigned>& xs);

    NodeId create_node(unsigned x, unsigned y);

    struct Node;
    std::vector<NodeId> col_ids_;
    std::vector<unsigned> sizes_;
    std::vector<Node> nodes_;
};

struct NodeMatrix::Node
{
    NodeId id;
    unsigned x, y;
    NodeId l, r, u, d;
    explicit Node(NodeId id_, unsigned x_, unsigned y_)
            : id(id_), x(x_), y(y_),
              l(id), r(id), u(id), d(id)
    {
    }
};

inline auto NodeMatrix::width() const -> unsigned { return col_ids_.size(); }
inline auto NodeMatrix::root_id() const -> NodeId { return 0; }
inline auto NodeMatrix::X(NodeId id) const -> unsigned { return nodes_[id].x; }
inline auto NodeMatrix::Y(NodeId id) const -> unsigned { return nodes_[id].y; }
inline auto NodeMatrix::S(NodeId id) const -> unsigned { return sizes_[X(id)]; }
inline auto NodeMatrix::C(NodeId id) const -> NodeId { return col_ids_[X(id)]; }
inline auto NodeMatrix::L(NodeId id) const -> NodeId { return nodes_[id].l; }
inline auto NodeMatrix::R(NodeId id) const -> NodeId { return nodes_[id].r; }
inline auto NodeMatrix::U(NodeId id) const -> NodeId { return nodes_[id].u; }
inline auto NodeMatrix::D(NodeId id) const -> NodeId { return nodes_[id].d; }