"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x20x20xbf16>) -> tensor<?x20x20xbf16>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x20x20xbf16>):
    %0 = "stablehlo.negate"(%arg1) : (tensor<?x20x20xbf16>) -> tensor<?x20x20xbf16>
    %1 = "stablehlo.exponential"(%0) : (tensor<?x20x20xbf16>) -> tensor<?x20x20xbf16>
    %2 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
    %3 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %4 = "stablehlo.reshape"(%3) : (tensor<i32>) -> tensor<1xi32>
    %5 = "stablehlo.constant"() <{value = dense<20> : tensor<1xi32>}> : () -> tensor<1xi32>
    %6 = "stablehlo.constant"() <{value = dense<20> : tensor<1xi32>}> : () -> tensor<1xi32>
    %7 = "stablehlo.concatenate"(%4, %5, %6) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %8 = "stablehlo.dynamic_broadcast_in_dim"(%2, %7) <{broadcast_dimensions = array<i64>}> : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %9 = "stablehlo.add"(%8, %1) : (tensor<?x20x20xbf16>, tensor<?x20x20xbf16>) -> tensor<?x20x20xbf16>
    %10 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
    %11 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %12 = "stablehlo.reshape"(%11) : (tensor<i32>) -> tensor<1xi32>
    %13 = "stablehlo.constant"() <{value = dense<20> : tensor<1xi32>}> : () -> tensor<1xi32>
    %14 = "stablehlo.constant"() <{value = dense<20> : tensor<1xi32>}> : () -> tensor<1xi32>
    %15 = "stablehlo.concatenate"(%12, %13, %14) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %16 = "stablehlo.dynamic_broadcast_in_dim"(%10, %15) <{broadcast_dimensions = array<i64>}> : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %17 = "stablehlo.divide"(%16, %9) : (tensor<?x20x20xbf16>, tensor<?x20x20xbf16>) -> tensor<?x20x20xbf16>
    "func.return"(%17) : (tensor<?x20x20xbf16>) -> ()
  }) : () -> ()
}) : () -> ()

