"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x2x3xf32>, tensor<?x2x3xf32>, tensor<?x2x3xf32>) -> tensor<?x2x3xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x2x3xf32>, %arg2: tensor<?x2x3xf32>, %arg3: tensor<?x2x3xf32>):
    %0 = "stablehlo.clamp"(%arg1, %arg2, %arg3) : (tensor<?x2x3xf32>, tensor<?x2x3xf32>, tensor<?x2x3xf32>) -> tensor<?x2x3xf32>
    "func.return"(%0) : (tensor<?x2x3xf32>) -> ()
  }) : () -> ()
}) : () -> ()

