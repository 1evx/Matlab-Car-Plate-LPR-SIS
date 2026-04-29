function results = runAllTests()
%RUNALLTESTS Convenience wrapper to run all MATLAB tests in the repo.

startup;
results = runtests({'tests/unit','tests/integration'});
disp(table(results));
end
