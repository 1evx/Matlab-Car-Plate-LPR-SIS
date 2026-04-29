function config = validateConfig(config)
%VALIDATECONFIG Merge user overrides into the default configuration.

defaults = defaultConfig();

if nargin < 1 || isempty(config)
    config = defaults;
    return;
end

config = mergeStructs(defaults, config);

requiredPaths = string(struct2cell(config.paths));
for i = 1:numel(requiredPaths)
    parentFolder = fileparts(requiredPaths(i));
    if strlength(parentFolder) > 0 && ~isfolder(parentFolder)
        mkdir(parentFolder);
    end
end
end

function output = mergeStructs(base, overrides)
output = base;
fields = fieldnames(overrides);

for i = 1:numel(fields)
    fieldName = fields{i};
    overrideValue = overrides.(fieldName);
    baseHasField = isfield(base, fieldName);
    if baseHasField
        baseValue = base.(fieldName);
    else
        baseValue = [];
    end

    if isstruct(overrideValue) && baseHasField && isstruct(baseValue) && isscalar(overrideValue) && isscalar(baseValue)
        output.(fieldName) = mergeStructs(baseValue, overrideValue);
    else
        output.(fieldName) = overrideValue;
    end
end
end
