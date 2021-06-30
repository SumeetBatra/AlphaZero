//
// Created by Sumeet Batra on 1/5/21.
//
#include <pybind11/embed.h>
#include <string>

namespace py = pybind11;


std::string push(std::string fen, std::string move) {
    /*
     * Make a move with python-chess and return the new board's fen
     */
    py::scoped_interpreter python;

    auto chess = py::module::import("chess");
    auto board = chess.attr("Board")(fen);
    auto chess_move = chess.attr("Move.from_uci")(move);
    auto new_board = chess.attr("push")(chess_move);
    return new_board.attr("fen")().cast<std::string>();
}