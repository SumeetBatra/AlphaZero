//
// Created by Sumeet Batra on 1/5/21.
//
#include <stdint.h>
//#include "mcts.h"
//#include "thc.h"
#include <stdio.h>
#include <string>
#include <vector>
#include <torch/script.h>
#include <torch/torch.h>

//void display_position( thc::ChessRules &cr, const std::string &description )
//{
//    std::string fen = cr.ForsythPublish();
//    std::string s = cr.ToDebugStr();
//    printf( "%s\n", description.c_str() );
//    printf( "FEN (Forsyth Edwards Notation) = %s\n", fen.c_str() );
//    printf( "Position = %s\n", s.c_str() );
//}

int main(int argc, char* argv[]){
    // testing features and debugging code
//    thc::ChessRules cr;
//    thc::Move mv;
//    printf("Size is %lu\n", sizeof(cr));
//    printf("Size of char is %lu\n", sizeof(char));
//
//    mv.TerseIn(&cr, "e2e4");
//    mv.TerseIn(&cr, "e7e6");
//    cr.PushMove(mv);
//    display_position(cr, "Pushed move e2e4");
//    printf("Move count is %d", cr.half_move_clock);
//
//    xt::xarray<double> test = {1.0, 2.0};
//    Node root = Node("abs",  -1);
//    MCTS mcts = MCTS(root);
//    Node child = root.select_leaf();
//    mcts.search();
    std::string s = "abs";
    torch::jit::script::Module module;
    return -1;
}