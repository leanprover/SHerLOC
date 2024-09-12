"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x2x3xf32>, tensor<?x2x3xf32>, tensor<?xf32>) -> tensor<?x2x3xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x2x3xf32>, %arg2: tensor<?x2x3xf32>, %arg3: tensor<?xf32>):
    %0 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %1 = "stablehlo.reshape"(%0) : (tensor<i32>) -> tensor<1xi32>
    %2 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = "stablehlo.constant"() <{value = dense<3> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "stablehlo.concatenate"(%1, %2, %3) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %5 = "stablehlo.dynamic_broadcast_in_dim"(%arg3, %4) <{broadcast_dimensions = array<i64: 0>}> : (tensor<?xf32>, tensor<3xi32>) -> tensor<?x2x3xf32>
    %6 = "stablehlo.clamp"(%arg1, %arg2, %5) : (tensor<?x2x3xf32>, tensor<?x2x3xf32>, tensor<?x2x3xf32>) -> tensor<?x2x3xf32>
    "func.return"(%6) : (tensor<?x2x3xf32>) -> ()
  }) : () -> ()
}) : () -> ()

