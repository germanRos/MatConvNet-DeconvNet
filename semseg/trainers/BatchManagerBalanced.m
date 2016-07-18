classdef BatchManagerBalanced < BatchManagerBase
   
    properties
        balanceStruct
    end
    
    methods
         function obj = BatchManagerBalanced(opts) 
            obj = obj@BatchManagerBase(opts);
            obj.mode = opts.mode;
            obj.balanceStruct = opts.balanceStruct;
            obj.numBatches = opts.maxNumBatches;
        end
        

        function batch = getBatch(obj, t)
            batch = [];
            
            % each row refers to a domain
            for i=1:size(obj.balanceStruct, 1)
                domain          = obj.balanceStruct(i, 1);
                num_imgs_domain = obj.balanceStruct(i, 2);
		
                % images available for this domain
                idxs = find(obj.imdb.images.set == domain);
                rand_idxs = idxs(randperm(numel(idxs), num_imgs_domain));
                batch(end+1:end+numel(rand_idxs)) = rand_idxs;
            end

            batch = batch(randperm(numel(batch)));
            
            obj.current = obj.current + 1;
            if(obj.current > obj.numBatches)
                obj.current = 1;
            end
        end
    end
    
end

