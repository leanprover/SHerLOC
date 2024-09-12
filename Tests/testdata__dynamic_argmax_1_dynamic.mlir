"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x4x5xf32>) -> tensor<?x5xi32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg6: tensor<i64>, %arg7: tensor<?x4x5xf32>):
    %18 = "func.call"(%arg6, %arg7) <{callee = @argmax}> : (tensor<i64>, tensor<?x4x5xf32>) -> tensor<?x5xi32>
    "func.return"(%18) : (tensor<?x5xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<?x4x5xf32>) -> tensor<?x5xi32>, sym_name = "argmax", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x4x5xf32>):
    %0 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %1 = "stablehlo.reshape"(%0) : (tensor<i32>) -> tensor<1xi32>
    %2 = "stablehlo.constant"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = "stablehlo.constant"() <{value = dense<5> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "stablehlo.concatenate"(%1, %2, %3) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %5 = "stablehlo.dynamic_iota"(%4) <{iota_dimension = 1 : i64}> : (tensor<3xi32>) -> tensor<?x4x5xi32>
    %6 = "stablehlo.constant"() <{value = dense<0xFF800000> : tensor<f32>}> : () -> tensor<f32>
    %7 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %8:2 = "stablehlo.reduce"(%arg1, %5, %6, %7) <{dimensions = array<i64: 1>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<f32>, %arg5: tensor<i32>):
      %9 = "stablehlo.compare"(%arg2, %arg4) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %10 = "stablehlo.compare"(%arg2, %arg2) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %11 = "stablehlo.or"(%9, %10) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %12 = "stablehlo.compare"(%arg2, %arg4) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %13 = "stablehlo.compare"(%arg3, %arg5) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %14 = "stablehlo.and"(%12, %13) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %15 = "stablehlo.or"(%11, %14) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %16 = "stablehlo.select"(%11, %arg2, %arg4) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %17 = "stablehlo.select"(%15, %arg3, %arg5) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
      "stablehlo.return"(%16, %17) : (tensor<f32>, tensor<i32>) -> ()
    }) : (tensor<?x4x5xf32>, tensor<?x4x5xi32>, tensor<f32>, tensor<i32>) -> (tensor<?x5xf32>, tensor<?x5xi32>)
    "func.return"(%8#1) : (tensor<?x5xi32>) -> ()
  }) : () -> ()
}) : () -> ()

