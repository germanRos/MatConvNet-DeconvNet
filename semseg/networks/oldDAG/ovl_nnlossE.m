function Y = vl_nnlossE(X,c,dzdy)
% VL_NNLOSS  CNN log-loss
%    Y = VL_NNLOSS(X, C) applies the the logistic loss to the data
%    X. X has dimension H x W x D x N, packing N arrays of W x H
%    D-dimensional vectors.
%
%    C contains the class labels, which should be integers in the range
%    1 to D. C can be an array with either N elements or with dimensions
%    H x W x 1 x N dimensions. In the fist case, a given class label is
%    applied at all spatial locations; in the second case, different
%    class labels can be specified for different locations.
%
%    DZDX = VL_NNLOSS(X, C, DZDY) computes the derivative DZDX of the
%    function projected on the output derivative DZDY.
%    DZDX has the same dimension as X.

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

% S = [ 1.0000
%     1.0000
%     2.0000
%     3.5000
%     7.5000
%     2.0000
%     5.5000
%    10.0000
%     7.0000
%    10.0000
%    14.0000
%    14.0000];

S = [ 3.8779    1.0000    1.1303    8.3688   15.3761    1.7578   96.8503    2.7390   19.8330   18.5157   52.8992];

[m_, c] = max(c, [], 3);
c(:,:,2,:) = S(c);

% no division by zero
X = X + 1e-4 ;
sz = [size(X,1) size(X,2) size(X,3) size(X,4)] ;

if numel(c) == sz(4)
  % one label per image
  c = reshape(c, [1 1 1 sz(4)]) ;
end
if size(c,1) == 1 & size(c,2) == 1
  c = repmat(c, [sz(1) sz(2)]) ;
end

% one label per spatial location
sz_ = [size(c,1) size(c,2) size(c,3) size(c,4)] ;
assert(isequal(sz_, [sz(1) sz(2) sz_(3) sz(4)])) ;
assert(sz_(3)==1 | sz_(3)==2) ;

% class c = 0 skips a spatial location
mass = single(c(:,:,1,:) > 0) ;
if sz_(3) == 2
  % the second channel of c (if present) is used as weights
  mass = mass .* c(:,:,2,:) ;
  c(:,:,2,:) = [] ;
end

% convert to indexes
c = c - 1 ;
c_ = 0:numel(c)-1 ;
c_ = 1 + ...
  mod(c_, sz(1)*sz(2)) + ...
  (sz(1)*sz(2)) * max(c(:), 0)' + ...
  (sz(1)*sz(2)*sz(3)) * floor(c_/(sz(1)*sz(2))) ;

%n = sz(1)*sz(2) ;
ke = (c(:,:,1,:) > 0) ;
n = sum(ke(:));

if nargin <= 2
  t = reshape(X(c_), [sz(1:2) 1 sz(4)]) ;
  Y = - sum(sum(sum(log(t) .* mass,1),2),4)/n ;
else
  Y_ = - bsxfun(@rdivide, bsxfun(@times, mass, dzdy), X)/n ;
  Y = Y_*0 ;
  Y(c_) = Y_(c_) ;
end
