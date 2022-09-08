H1 = 0.1;
Point(1) = {0, 0, 0, H1};
//+
Point(2) = {0, 1, 0, H1};
//+
Point(3) = {25, 1, 0, H1};
//+
Point(4) = {25, 0, 0, H1};
//+
Point(5) = {0, 1, 1.0, H1};
//+
Point(6) = {25, 1, 1.0, H1};
//+
Point(7) = {25, 0, 1.0, H1};
//+
Point(8) = {0, 0, 1.0, H1};
//+
Line(1) = {2, 1};
//+
Line(2) = {1, 8};
//+
Line(3) = {8, 5};
//+
Line(4) = {5, 2};
//+
Line(5) = {2, 3};
//+
Line(6) = {3, 6};
//+
Line(7) = {6, 5};
//+
Line(8) = {4, 3};
//+
Line(9) = {4, 7};
//+
Line(10) = {7, 6};
//+
Line(11) = {7, 8};
//+
Line(12) = {4, 1};
//+
Curve Loop(1) = {9, 10, -6, -8};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {2, 3, 4, 1};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {12, 2, -11, -9};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {5, 6, 7, 4};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {3, -7, -10, 11};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {12, -1, 5, -8};
//+
Plane Surface(6) = {6};
//+
Surface Loop(1) = {1, 3, 6, 2, 5, 4};
//+
Volume(1) = {1};

