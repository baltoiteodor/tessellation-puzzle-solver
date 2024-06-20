#include <pybind11/pybind11.h>
#include <iostream>
#include <pybind11/stl.h>
#include <ctime>
#include <cstdlib>

namespace py = pybind11;


// Opted for monolithic file as setting up the other files and cmake file would be very time-consuming.
// Good extension to this project would be to make the C++ part prettier.


// Class for representing the double linked list.
class NodeMatrix {
public:
    using NodeId = int;

    explicit NodeMatrix(std::vector<std::vector<int>> rows, int wid): columns_(wid), sizes_(wid) {
        // Set x - header column, and y - row number to big values for the root.
        NodeId root = new_node(~0, ~0);

        // Root has Id 0.
        assert(root == 0);

        for (auto x = 0u; x < columns_.size(); ++x) {
            NodeId id = new_node(x, ~0);
            columns_[x] = id;

            nodes_[id].right = root;
            nodes_[id].left = L(root);
            nodes_[L(root)].right = id;
            nodes_[root].left = id;

        }

        for (auto y = 0u; y < rows.size(); ++y) {
            add_row(y, rows[y]);
        }
    }

    void cover_column(NodeId c) {
        c = C(c);
        nodes_[L(c)].right = R(c);
        nodes_[R(c)].left = L(c);
        for (NodeId i = D(c); i != c; i = D(i)) {
            for (NodeId j = R(i); j != i; j = R(j)) {
                nodes_[U(j)].down = D(j);
                nodes_[D(j)].up = U(j);
                --sizes_[X(j)];
            }
        }
    }

    void uncover_column(NodeId c) {
        c = C(c);
        for (NodeId i = U(c); i != c; i = U(i)) {
            for (NodeId j = L(i); j != i; j = L(j)) {
                nodes_[U(j)].down = j;
                nodes_[D(j)].up = j;
                ++sizes_[X(j)];
            }
        }
        nodes_[L(c)].right = c;
        nodes_[R(c)].left = c;
    }

    inline unsigned width() const { return columns_.size(); }
    inline unsigned X(NodeId id) const { return nodes_[id].x; }
    inline unsigned Y(NodeId id) const { return nodes_[id].y; }
    inline unsigned S(NodeId id) const { return sizes_[X(id)]; }
    inline NodeId C(NodeId id) const { return columns_[X(id)]; }
    inline NodeId L(NodeId id) const { return nodes_[id].left; }
    inline NodeId R(NodeId id) const { return nodes_[id].right; }
    inline NodeId U(NodeId id) const { return nodes_[id].up; }
    inline NodeId D(NodeId id) const { return nodes_[id].down; }

private:
    void add_row(unsigned y, const std::vector<int>& xs) {
        NodeId first_id = 0;
        for (auto x : xs) {
            NodeId id = new_node(x, y);
            nodes_[id].down = C(id);
            nodes_[id].up = U(C(id));
            nodes_[U(C(id))].down = id;
            nodes_[C(id)].up = id;
            ++sizes_[x];
            if (first_id == 0) {
                first_id = id;
            }
            else {
                nodes_[id].right = first_id;
                nodes_[id].left = L(first_id);
                nodes_[L(first_id)].right = id;
                nodes_[first_id].left = id;
            }
        }
    }

    NodeId new_node(unsigned x, unsigned y) {
        assert(x <= width() || x == ~0u);
        unsigned id = nodes_.size();
        nodes_.emplace_back(id, x, y);
        return id;
    }

    struct Node {
        NodeId id;
        unsigned x, y;
        NodeId left, right, up, down;
        // Initially node is connected to itself at all points.
        explicit Node(NodeId id_, unsigned x_, unsigned y_)
                : id(id_), x(x_), y(y_),
                  left(id), right(id), up(id), down(id)
        {
        }
    };

    std::vector<NodeId> columns_;
    std::vector<unsigned> sizes_;
    std::vector<Node> nodes_;
};

//
// Here is where we interact with the C++ code from python and consists of the DLX algorithm.
//

class DLXSolver{
public:

    struct SearchState {
        std::vector<int> stack;
        bool stopped = false;
    };

    DLXSolver(std::vector<std::vector<int>> rows, int width): rows_(rows), width_(width), matrix_(rows, width){};

    void printer() {
        std::cout << rowsNum_ << " That is how many rows there are.\n";
    }

    std::vector<std::vector<int>> getRows() {
        return rows_;
    }

    int getWidth() {
        return width_;
    }

    std::vector<std::vector<int>> solve(int maxSol) {
        std::vector<int> solution;
        SearchState state = SearchState();
        search(solution, state, maxSol);
        return solutions_;
    }

    void search(std::vector<int> &solution, SearchState &state, int maxSol) {
        if (state.stopped) {
            return;
        }

        // Assign h 0, a.k.a. root, and verify if root property holds.
        // If reached the max number of solutions, return.
        auto h = 0;
        if (R(h) == h) {
            solution = state.stack;
            solutions_.push_back(solution);
            if(solutions_.size() >= maxSol) {
                state.stopped = true;
            }
            return;
        }

        auto cs = std::vector<NodeId>();
        for (auto j = R(h); j != h; j = R(j)) {
            if (!cs.empty() && S(j) < S(cs[0])) {
                cs.clear();
            }
            if (cs.empty() || S(j) == S(cs[0])) {
                cs.push_back(j);
            }
        }
        assert(!cs.empty());
        if (S(cs[0]) < 1) {
            return;
        }

        auto c = cs[0];
        // Always go random column.
        auto randint = std::rand() % cs.size();
        c = cs[randint];

        cover_column(c);
        for (auto r = D(c); r != c; r = D(r)) {
            state.stack.push_back(Y(r));
            for (auto j = R(r); j != r; j = R(j))
                cover_column(j);
            search(solution, state, maxSol);
            for (auto j = L(r); j != r; j = L(j))
                uncover_column(j);
            state.stack.pop_back();
        }
        uncover_column(c);
    }

//    double calculateSim(cv::Mat img1, cv::Mat img2) {
//        cv::Mat img1_gray, img2_gray;
//        cv::cvtColor(img1, img1_gray, cv::COLOR_BGR2GRAY);
//        cv::cvtColor(img2, img2_gray, cv::COLOR_BGR2GRAY);
//
//        cv::Mat ssim_map;
//        cv::Mat temp1, temp2, temp3;
//
//        const double C1 = 6.5025, C2 = 58.5225;
//
//        cv::Mat I1, I2;
//        img1_gray.convertTo(I1, CV_32F);
//        img2_gray.convertTo(I2, CV_32F);
//
//        cv::Mat I1_2 = I1.mul(I1);
//        cv::Mat I2_2 = I2.mul(I2);
//        cv::Mat I1_I2 = I1.mul(I2);
//
//        cv::GaussianBlur(I1, temp1, cv::Size(11, 11), 1.5);
//        cv::GaussianBlur(I2, temp2, cv::Size(11, 11), 1.5);
//
//        cv::Mat mu1 = temp1;
//        cv::Mat mu2 = temp2;
//
//        cv::Mat mu1_2 = mu1.mul(mu1);
//        cv::Mat mu2_2 = mu2.mul(mu2);
//        cv::Mat mu1_mu2 = mu1.mul(mu2);
//
//        cv::GaussianBlur(I1_2, temp1, cv::Size(11, 11), 1.5);
//        cv::GaussianBlur(I2_2, temp2, cv::Size(11, 11), 1.5);
//        cv::GaussianBlur(I1_I2, temp3, cv::Size(11, 11), 1.5);
//
//        cv::Mat sigma1_2 = temp1 - mu1_2;
//        cv::Mat sigma2_2 = temp2 - mu2_2;
//        cv::Mat sigma12 = temp3 - mu1_mu2;
//
//        cv::Mat t1, t2, t3;
//
//        t1 = 2 * mu1_mu2 + C1;
//        t2 = 2 * sigma12 + C2;
//        t3 = t1.mul(t2);
//
//        t1 = mu1_2 + mu2_2 + C1;
//        t2 = sigma1_2 + sigma2_2 + C2;
//        t1 = t1.mul(t2);
//
//        cv::divide(t3, t1, ssim_map);
//
//        cv::Scalar mssim = cv::mean(ssim_map);
//        return mssim[0];
//    }


private:
    NodeMatrix matrix_;
    std::vector<std::vector<int>> solutions_;
    int rowsNum_ = 0;
    std::vector<std::vector<int>> rows_;
    int width_ = 0;



    using NodeId = NodeMatrix::NodeId;
    void cover_column(NodeId id) { matrix_.cover_column(id); }
    void uncover_column(NodeId id) { matrix_.uncover_column(id); }
    unsigned Y(NodeId id) { return matrix_.Y(id); }
    unsigned S(NodeId id) { return matrix_.S(id); }
    NodeId L(NodeId id) { return matrix_.L(id); }
    NodeId R(NodeId id) { return matrix_.R(id); }
    NodeId U(NodeId id) { return matrix_.U(id); }
    NodeId D(NodeId id) { return matrix_.D(id); }
};

PYBIND11_MODULE(DLXCPP, handle) {
handle.doc() = "Module DLX docs.";
//    handle.def("add_cpp", &add);

py::class_<DLXSolver>(
        handle, "DLXCPPSolver"
)
.def(py::init<std::vector<std::vector<int>>, int>())
.def("printer", &DLXSolver::printer)
.def("getRows", &DLXSolver::getRows)
.def("getWidth", &DLXSolver::getWidth)
.def("solve", &DLXSolver::solve)
//        .def("calculate_ssim", [](DLXSolver &self, py::array_t<unsigned char> img1, py::array_t<unsigned char> img2) {
//            cv::Mat mat_img1 = numpy_to_mat(img1);
//            cv::Mat mat_img2 = numpy_to_mat(img2);
//            return self.calculate_ssim(mat_img1, mat_img2);
//        })
;
}