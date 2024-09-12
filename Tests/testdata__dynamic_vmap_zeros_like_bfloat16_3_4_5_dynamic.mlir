"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x3x4x5xbf16>) -> tensor<?x3x4x5xbf16>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x3x4x5xbf16>):
    %0 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
    %1 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %2 = "stablehlo.reshape"(%1) : (tensor<i32>) -> tensor<1xi32>
    %3 = "stablehlo.constant"() <{value = dense<3> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "stablehlo.constant"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
    %5 = "stablehlo.constant"() <{value = dense<5> : tensor<1xi32>}> : () -> tensor<1xi32>
    %6 = "stablehlo.concatenate"(%2, %3, %4, %5) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %7 = "stablehlo.dynamic_broadcast_in_dim"(%0, %6) <{broadcast_dimensions = array<i64>}> : (tensor<bf16>, tensor<4xi32>) -> tensor<?x3x4x5xbf16>
    "func.return"(%7) : (tensor<?x3x4x5xbf16>) -> ()
  }) : () -> ()
}) : () -> ()

