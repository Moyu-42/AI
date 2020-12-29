valid((_,Col)):-
    Range = [1,2,3,4,5,6,7,8],
    member(Col,Range).

valid_board([]).
valid_board([Head|Tail]):- valid(Head),valid_board(Tail).

cols([],[]).
cols([(_,Col)|QueensTail],[Col|ColsTail]):-
    cols(QueensTail,ColsTail).
    
main_diag([],[]).
main_diag([(Row,Col)|QueensTail],[Diagonal|DiagonalsTail]):-
    Diagonal is Col - Row,
    main_diag(QueensTail,DiagonalsTail).
    
diag([],[]).
diag([(Row,Col)|QueensTail],[Diagonal|DiagonalsTail]):-
    Diagonal is Col + Row,
    diag(QueensTail,DiagonalsTail).

eight_queens(Board) :-
    Board = [(1, _), (2, _), (3, _), (4, _), (5, _), (6, _), (7, _), (8, _)],
    valid_board(Board),
    
    cols(Board,Cols),
    main_diag(Board,Main_diag),
    diag(Board,Diag),
    
    fd_all_different(Cols),
    fd_all_different(Main_diag),
    fd_all_different(Diag).
