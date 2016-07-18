function Y = vl_nnpadder(X, padd, dzdy)
	if nargin < 3;
		top = gpuArray.zeros(padd(1), size(X, 2), size(X, 3), size(X,4), 'single');
		dow = gpuArray.zeros(padd(2), size(X, 2), size(X, 3), size(X,4), 'single');
		lef = gpuArray.zeros(size(X, 1)+padd(1)+padd(2), padd(3), size(X, 3), size(X,4), 'single');
		rig = gpuArray.zeros(size(X, 1)+padd(1)+padd(2), padd(4), size(X, 3), size(X,4), 'single');
		Y = [lef, [top; X; dow], rig];

	% backward mode
	else
		% ojo a esto!
		Y=X(padd(1):end-padd(2), padd(3):end-padd(4)+1, :, :);
		%disp('ook');
		%size(Y)
	end


end
