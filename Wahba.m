%{
  /* Solve Wahba's problem using Kabsch algorithm: 
   * arg_min(Q) \sum_k || w_k - Q v_k ||^2
   * Q is rotation matrix, w_k and v_k are vectors to be aligned.
   * Solution:  Denote B = \sum_k w_k v_k^T
   * Decompose B = U * S * V^T 
   * Then Q = U * M * V^T, where M = diag[ 1 1 det(U) det(V) ] 
   * Refs: http://journals.iucr.org/a/issues/1976/05/00/a12999/a12999.pdf 
   *       http://www.control.auc.dk/~tb/best/aug23-Bak-svdalg.pdf */
%}
clc;
Arrow = [
    0 0 0 1 0 0 0;
    0 0 1 1 1 0 0; 
    0 1 0 1 0 1 0; 
    0 0 0 1 0 0 0;
    0 0 0 1 0 0 0; 
    0 0 0 0 0 0 0;
    0 0 0 0 0 0 0;
    ]; 
angle = -45; 


figure(1); imagesc(Arrow); colormap('gray');

ArrowRotated = imrotate( Arrow, angle, 'bilinear'); 
figure(2); imagesc(ArrowRotated); colormap('gray'); 
w = [1 0 0]'; 
v = [0 0 1]';
B = w * v'; 
[U S V] = svd(B); 
M = diag([1 1 1], 0); 
Q = U * M * V'; 
w
v
v_rotated = Q * v; 
v_rotated