classdef BatchManagerBase < handle
    % BatchManager is the abstract class for defining batch policies 
    
    properties(GetAccess='protected', SetAccess='protected')
        numBatches = 0;
        mode = '';
        current = 1;
        imdb = struct();
    end
    
    methods

        function obj = BatchManagerBase(opts)
         	obj = obj@handle(); 
                
		if(nargin ~= 0)
			obj.mode = opts.mode;
            		obj.imdb = opts.imdb;
		end
        end
       
        
        function numBatches = getNumBatches(obj)
            numBatches = obj.numBatches;
        end
        
        function batch = getBatch(obj, t)
            batch = [];
        end
    end
    
end

