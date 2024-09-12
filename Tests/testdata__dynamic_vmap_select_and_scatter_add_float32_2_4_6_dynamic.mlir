"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x1x3x5xf32>, tensor<?x2x4x6xf32>) -> tensor<?x2x4x6xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x1x3x5xf32>, %arg2: tensor<?x2x4x6xf32>):
    %0 = "stablehlo.constant"() <{value = dense<0xFF800000> : tensor<f32>}> : () -> tensor<f32>
    %1 = "stablehlo.pad"(%arg2, %0) <{edge_padding_high = array<i64: 0, 0, 0, 0>, edge_padding_low = array<i64: 0, 0, 0, 0>, interior_padding = array<i64: 0, 0, 0, 0>}> : (tensor<?x2x4x6xf32>, tensor<f32>) -> tensor<?x2x4x6xf32>
    %2 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %3 = "stablehlo.select_and_scatter"(%1, %arg1, %2) <{window_dimensions = array<i64: 1, 2, 2, 2>}> ({
    ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):
      %22 = "stablehlo.compare"(%arg5, %arg6) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "stablehlo.return"(%22) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %21 = "stablehlo.add"(%arg3, %arg4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%21) : (tensor<f32>) -> ()
    }) : (tensor<?x2x4x6xf32>, tensor<?x1x3x5xf32>, tensor<f32>) -> tensor<?x2x4x6xf32>
    %4 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %5 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %6 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %7 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %8 = "stablehlo.concatenate"(%4, %5, %6, %7) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %9 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %10 = "stablehlo.reshape"(%9) : (tensor<i32>) -> tensor<1xi32>
    %11 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %12 = "stablehlo.constant"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
    %13 = "stablehlo.constant"() <{value = dense<6> : tensor<1xi32>}> : () -> tensor<1xi32>
    %14 = "stablehlo.concatenate"(%10, %11, %12, %13) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %15 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %16 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %17 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %18 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %19 = "stablehlo.concatenate"(%15, %16, %17, %18) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %20 = "stablehlo.real_dynamic_slice"(%3, %8, %14, %19) : (tensor<?x2x4x6xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<?x2x4x6xf32>
    "func.return"(%20) : (tensor<?x2x4x6xf32>) -> ()
  }) : () -> ()
}) : () -> ()

