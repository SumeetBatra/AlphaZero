//
// Created by Sumeet Batra on 1/5/21.
//
#include <stdint.h>
#include "mcts.h"
#include "thc.h"
#include <stdio.h>
#include <string>
#include <vector>

void display_position( thc::ChessRules &cr, const std::string &description )
{
    std::string fen = cr.ForsythPublish();
    std::string s = cr.ToDebugStr();
    printf( "%s\n", description.c_str() );
    printf( "FEN (Forsyth Edwards Notation) = %s\n", fen.c_str() );
    printf( "Position = %s\n", s.c_str() );
}

int main(int argc, char* argv[]){
    // testing features and debugging code
    thc::ChessRules cr;
    display_position(cr, "Initial position");

    xt::xarray<double> test = {1.0, 2.0};
    Node root = Node("abs",  1);
    MCTS mcts = MCTS(root);
    Node child = root.select_leaf();
//    mcts.search();
    return -1;
}