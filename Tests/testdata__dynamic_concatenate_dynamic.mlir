"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<i64>, tensor<?x?x5xf32>) -> tensor<?x?x5xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<?x?x5xf32>):
    %0 = "stablehlo.concatenate"(%arg2, %arg2) <{dimension = 0 : i64}> : (tensor<?x?x5xf32>, tensor<?x?x5xf32>) -> tensor<?x?x5xf32>
    "func.return"(%0) : (tensor<?x?x5xf32>) -> ()
  }) : () -> ()
}) : () -> ()

