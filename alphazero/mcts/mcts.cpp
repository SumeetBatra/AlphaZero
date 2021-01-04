//
// Created by Sumeet Batra on 1/3/21.
//
#include <string>
#include <vector>
#include <tuple>
#include "mcts.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xrandom.hpp"

MCTS::MCTS(Node &root_node) : root(root_node){}

void MCTS::set_root(Node &root_node) {
    root = root_node;
}

void MCTS::backprop(Node &leaf, double val) {
    leaf.set_visit_count(leaf.get_visit_count() + 1);
    Node node = leaf;
    while(node.parent != NULL) {
        node.parent->child_W(node.move) += val + 1; // +1 to counteract virtual loss
        node.parent->child_Q(node.move) = node.parent->child_W(node.move) / node.parent->child_N(node.move);
        node = *node.parent;
        val = -val;
    }
}

auto MCTS::play(double temperature) {
    // struct contains the values we wish to return
    struct ret_vals{
        std::string move;
        Node child;
        std::tuple <std::string, xt::xarray<double>, double> datapoint; // (board_fen, logits, reward) tuple
    };
    xt::xarray<double> logits = this->get_root().child_N / this->root.get_visit_count();

    int action = -1;
    if(temperature == 0) {
        xt::xarray<int> action_inds = xt::argmax(logits);
        action = action_inds(0);  // TODO: make sure this is correct
        xt::xarray<double> logits = xt::zeros_like(logits);
        logits(action) = 1.0;
    }else{
        xt::xarray<double> temp = {temperature};
        logits = xt::pow(logits, temp);
        xt::xarray<double> action_ind = xt::random::choice(logits, 1);
        action = action_ind(0);
    }
    return ret_vals{
        this->get_root().legal_moves[action],
        this->get_root().children[action],
        std::tuple<std::string, xt::xarray<double>, double> {this->root.board_fen, logits, -1.0} // -1 is a placeholder since we don't know the reward yet
    };
}


Node::Node(std::string board_fen, int move, Node* parent) : board_fen(board_fen), move(move), parent(parent) {}

void Node::set_visit_count(uint16_t visits) {
    this->parent->child_N(move) = visits;
}

void Node::set_total_value(int value) {
    this->parent->child_W(move) = value;
}

xt::xarray<double> Node::puct(double cpuct) {
    // see https://ai.stackexchange.com/questions/25451/how-does-alphazeros-mcts-work-when-starting-from-the-root-node for why U is calculated this way
    xt::xarray<double> U = cpuct * this->child_P * xt::sqrt(xt::xarray<double>{double(this->get_visit_count())}) / (1.0 + this->parent->child_N);
    return this->child_Q + U;
}

Node Node::best_child() {
    auto move_inds = xt::argmax(this->puct());
    double move_idx = move_inds(0);
    bool child_exists = this->children.count(move_idx);
    if(child_exists == 1) {
        return this->children[move_idx];
    }else{
        //TODO: Need to interface with Python chess library here to get the new board's fen
        std::string new_board_fen = "";
        return Node(new_board_fen, move_idx, this);;
    }
}

Node Node::select_leaf() {
    Node node = *this;
    while(node.visited) {
        // virtual loss
        // TODO: acquire lock here before modifying the next two attributes
        node.set_visit_count(node.get_visit_count() + 1);
        node.set_total_value(node.get_total_value() - 1);

        node = node.best_child();
    }
    return node;
}

auto Node::action_value() {
    struct ret_vals{
        xt::xarray<double> p_acts;
        double val;
    };
    //TODO: Figure out how to implement this and get values back from the neural network
    //TODO: Will need to interface with python code here

    //placeholder
    xt::xarray<double> p_acts_raw = xt::ones<double>({4672});
    xt::xarray<double> p_acts = p_acts_raw / xt::sum(p_acts_raw);
    return std::tuple<xt::xarray<double>, double> (p_acts, -1.0);
}

double Node::expand(bool root) {
    this->visited = true;
    xt::xarray<double> p_acts;
    double val;
    tie(p_acts, val) = this->action_value();

    if(root){
        //TODO: figure out how to use PyTorch here
    }
    //TODO: will need to interact with python here to get the encoded actions
    return val;
}
