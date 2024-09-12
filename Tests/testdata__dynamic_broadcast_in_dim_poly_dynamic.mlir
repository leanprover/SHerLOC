"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x1x4xf32>) -> tensor<?x?x4xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x1x4xf32>):
    %0 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
    %1 = "stablehlo.multiply"(%arg0, %0) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %2 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %3 = "stablehlo.reshape"(%2) : (tensor<i32>) -> tensor<1xi32>
    %4 = "stablehlo.convert"(%1) : (tensor<i64>) -> tensor<i32>
    %5 = "stablehlo.reshape"(%4) : (tensor<i32>) -> tensor<1xi32>
    %6 = "stablehlo.constant"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
    %7 = "stablehlo.concatenate"(%3, %5, %6) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %8 = "stablehlo.dynamic_broadcast_in_dim"(%arg1, %7) <{broadcast_dimensions = array<i64: 0, 1, 2>}> : (tensor<?x1x4xf32>, tensor<3xi32>) -> tensor<?x?x4xf32>
    "func.return"(%8) : (tensor<?x?x4xf32>) -> ()
  }) : () -> ()
}) : () -> ()

