function Y = vl_nnfrozennet(X, net, params, dzdy)
	if nargin == 3
 		input(1).x = X;
    		input(1).name = 'data';
    		input(2).name = 'label';
    		input(2).x = gpuArray.zeros(1,1,1,1, 'single');

		opts2.cudaKernel1 = params.opts.cudaKernel1;
		opts2.conserveMemory = params.opts.conserveMemory;
		res = params.dagnn_fcn(net, input, [], [], [], 'disableDropout', false, opts2);
		
        
        Y = res(end).x;
     
        
	else
		Y = gpuArray.zeros(size(X), 'single');
	end
end
