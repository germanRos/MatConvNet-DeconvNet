function [ y ] = vl_nnconcatMultiFeat(inputs, dim, numFeats, dzdy)
%VL_NNCONCAT Concatenate multiple inputs

if nargin < 4, dzdy = []; end;

if isempty(dzdy) 
%     for i=1:numel(inputs)
%        disp(size(inputs{i})); 
%     end
  y = cat(dim, inputs{:});
else 
  y = cell(1, numel(numFeats));
  accum = 1;
  for i=1:numel(numFeats)
      switch dim
      	case 1
      	  y{i} = dzdy(accum:accum+numFeats(i)-1, :, :, :);
      	case 2
      	  y{i} = dzdy(:, accum:accum+numFeats(i)-1, :, :);
      	case 3
      	  y{i} = dzdy(:, :, accum:accum+numFeats(i)-1, :);
      end

      accum = accum + numFeats(i);
  end

end

