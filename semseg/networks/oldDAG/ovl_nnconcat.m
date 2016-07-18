function [ y ] = vl_nnconcat( inputs, dim, scale, crop, dzdy )
%VL_NNCONCAT Concatenate multiple inputs
if nargin < 2, dim = 3, scale=1, crop=[0,0,0,0]; end;
if nargin < 5, dzdy = []; end;

if isempty(dzdy)
  % we assume inputs{1} is always the smallest one
  X1 = imresize(inputs{1}, scale, 'cubic');
  % cropping
  I1 = X1(crop(1)+1:end-crop(2), crop(3)+1:end-crop(4), :, :);
  y = cat(dim, I1, inputs{2:end});
else
  numdiv = numel(inputs);
  insz = size(inputs{1});
  outsz = size(dzdy);
  %assert(outsz(dim) == insz(dim)*numdiv);
  
  y = cell(1, numdiv);
  divs = 1:insz(dim):(outsz(dim) + 1);  
  % especial treatment for the first channels due to re-scaling and
  % croppings.
  dd = [];
  lims = divs(1):divs(2)-1;
  switch dim
      case 1
        dd = dzdy(lims, :, :, :);
      case 2
        dd = dzdy(:, lims, :, :);
      case 3
        dd = dzdy(:, :, lims, :);
  end  
  
  res_dzdy1 = imresize(dd, 1/scale);
  y{1} = res_dzdy1;
    
  % standard way for the remaining channels...
  for di = 2
    lims = divs(di);
    switch dim
      case 1
        y{di} = dzdy(lims:end, :, :, :);
      case 2
        y{di} = dzdy(:, lims:end, :, :);
      case 3
        y{di} = dzdy(:, :, lims:end, :);
    end
  end
end


