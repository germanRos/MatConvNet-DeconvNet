classdef BatchManagerSeq < BatchManagerBase
    % BatchManager for sequential application
    
    properties
       domains = [];
       bsize = 0;
       data = [];
    end
    
    methods
        function obj = BatchManagerSeq(opts) 
            obj = obj@BatchManagerBase(opts);
            obj.mode = opts.mode;
            obj.domains = opts.domains;
            obj.bsize = opts.bsize;
            
            cellx = arrayfun(@(a) obj.imdb.images.set==a, opts.domains, 'UniformOutput', false);
            obj.data = find(any(cell2mat(cellx'), 1));
            obj.numBatches = ceil(numel(obj.data) / obj.bsize);
        end
        

        function batch = getBatch(obj, t)  
            batch = obj.data((t-1)*obj.bsize+1:min(t*obj.bsize, numel(obj.data)));
            obj.current = obj.current + 1;
        end
        
        
    end
    
end

