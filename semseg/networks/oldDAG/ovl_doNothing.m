function [Y] = vl_doNothing(inputs, selector, dzdy)
	% forward mode
	
	if nargin < 3
		Y = inputs{selector};
      	
	% backward mode
	else
		Y = cell(1, numel(inputs));
        Y{1} = dzdy;
        Y{2} = gpuArray.zeros(size(inputs{2}), 'single');
       
	end

end
