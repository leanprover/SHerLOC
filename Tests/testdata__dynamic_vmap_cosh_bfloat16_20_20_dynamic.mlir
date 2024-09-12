"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x20x20xbf16>) -> tensor<?x20x20xbf16>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x20x20xbf16>):
    %0 = "stablehlo.constant"() <{value = dense<20> : tensor<1xi32>}> : () -> tensor<1xi32>
    %1 = "stablehlo.constant"() <{value = dense<5.000000e-01> : tensor<f32>}> : () -> tensor<f32>
    %2 = "stablehlo.convert"(%arg1) : (tensor<?x20x20xbf16>) -> tensor<?x20x20xf32>
    %3 = "stablehlo.get_dimension_size"(%2) <{dimension = 0 : i64}> : (tensor<?x20x20xf32>) -> tensor<i32>
    %4 = "stablehlo.reshape"(%3) : (tensor<i32>) -> tensor<1xi32>
    %5 = "stablehlo.concatenate"(%4, %0, %0) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %6 = "stablehlo.dynamic_broadcast_in_dim"(%1, %5) <{broadcast_dimensions = array<i64>}> : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %7 = "stablehlo.log"(%6) : (tensor<?x20x20xf32>) -> tensor<?x20x20xf32>
    %8 = "stablehlo.add"(%2, %7) : (tensor<?x20x20xf32>, tensor<?x20x20xf32>) -> tensor<?x20x20xf32>
    %9 = "stablehlo.exponential"(%8) : (tensor<?x20x20xf32>) -> tensor<?x20x20xf32>
    %10 = "stablehlo.subtract"(%7, %2) : (tensor<?x20x20xf32>, tensor<?x20x20xf32>) -> tensor<?x20x20xf32>
    %11 = "stablehlo.exponential"(%10) : (tensor<?x20x20xf32>) -> tensor<?x20x20xf32>
    %12 = "stablehlo.add"(%9, %11) : (tensor<?x20x20xf32>, tensor<?x20x20xf32>) -> tensor<?x20x20xf32>
    %13 = "stablehlo.convert"(%12) : (tensor<?x20x20xf32>) -> tensor<?x20x20xbf16>
    "func.return"(%13) : (tensor<?x20x20xbf16>) -> ()
  }) : () -> ()
}) : () -> ()

