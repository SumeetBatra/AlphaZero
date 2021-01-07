//
// Created by Sumeet Batra on 1/3/21.
//

#ifndef ALPHAZERO_MCTS_H
#define ALPHAZERO_MCTS_H

#include <string>
#include <vector>
#include <map>
#include "xtensor/xarray.hpp"


class Node{
private:
    Node* parent;

    std::map<int, Node> children;

    xt::xarray<uint16_t> child_N;
    xt::xarray<double> child_W;
    xt::xarray<double> child_Q;
    xt::xarray<double> child_P;

    xt::xarray<double> puct(double cpuct=3.0);
    auto action_value(); // TODO: Figure out what to do with model
    bool is_terminal();

    double expand(bool root=false); // TODO: Figure out what to do with model


public:
    // default constructor
    Node();
    // constructor
    Node(std::string board_fen, int move, Node* parent=NULL);

    friend class MCTS;

    std::vector<std::string> board_fen_stack;
    std::vector<std::string> legal_moves;

    uint16_t repetitions;
    uint16_t move;
    bool visited;
    bool color;
    std::string board_fen;

    int get_visit_count(){return (parent != NULL) ? parent->child_N[move] : 1;}
    void set_visit_count(uint16_t visits);
    double get_total_value(){return (parent != NULL) ? parent->child_W[move] : 1;}
    void set_total_value(int value);

    Node best_child();
    Node select_leaf();

};

class MCTS{
private:
    Node root;
    void backprop(Node &leaf, double val);

public:
    MCTS(Node &root_node);  //TODO: Figure out what to do with the neural network

    Node get_root(){return root;}
    void set_root(Node &root_node);
    void search();
    auto play(double temperature);
};

#endif //ALPHAZERO_MCTS_H
