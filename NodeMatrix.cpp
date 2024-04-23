// Nodes in the matrix will be represented by ints, each node object will have a unique int id.
class NodeMatrix {
public:
    NodeMatrix(std::vector<std::vector<int>>rows, int width): _columnIds(width), _sizeIds(width) {}
private:
    struct Node;
    std::vector<int> _columnIds;
    std::vector<int> _sizeIds;
    // List of all Nodes. Will map from ids to Nodes for easier and small-size handling.
    std::vector<Node> _nodes;
};

struct NodeMatrix::Node {
    int id;
    int x, y;
    int left, right, up, down;
    // Initialise node: First all neighbours are same node as it might be first.
    explicit Node(int _id, int _x, int _y): id(_id), x(_x), y(_y), left(_id), right(_id), up(_id), down(_id) {}
};