"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x15xf32>) -> tensor<?xi32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg6: tensor<i64>, %arg7: tensor<?x15xf32>):
    %17 = "func.call"(%arg6, %arg7) <{callee = @argmin}> : (tensor<i64>, tensor<?x15xf32>) -> tensor<?xi32>
    "func.return"(%17) : (tensor<?xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<?x15xf32>) -> tensor<?xi32>, sym_name = "argmin", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x15xf32>):
    %0 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %1 = "stablehlo.reshape"(%0) : (tensor<i32>) -> tensor<1xi32>
    %2 = "stablehlo.constant"() <{value = dense<15> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = "stablehlo.concatenate"(%1, %2) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %4 = "stablehlo.dynamic_iota"(%3) <{iota_dimension = 1 : i64}> : (tensor<2xi32>) -> tensor<?x15xi32>
    %5 = "stablehlo.constant"() <{value = dense<0x7F800000> : tensor<f32>}> : () -> tensor<f32>
    %6 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %7:2 = "stablehlo.reduce"(%arg1, %4, %5, %6) <{dimensions = array<i64: 1>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<f32>, %arg5: tensor<i32>):
      %8 = "stablehlo.compare"(%arg2, %arg4) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %9 = "stablehlo.compare"(%arg2, %arg2) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %10 = "stablehlo.or"(%8, %9) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %11 = "stablehlo.compare"(%arg2, %arg4) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %12 = "stablehlo.compare"(%arg3, %arg5) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %13 = "stablehlo.and"(%11, %12) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %14 = "stablehlo.or"(%10, %13) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %15 = "stablehlo.select"(%10, %arg2, %arg4) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %16 = "stablehlo.select"(%14, %arg3, %arg5) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
      "stablehlo.return"(%15, %16) : (tensor<f32>, tensor<i32>) -> ()
    }) : (tensor<?x15xf32>, tensor<?x15xi32>, tensor<f32>, tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>)
    "func.return"(%7#1) : (tensor<?xi32>) -> ()
  }) : () -> ()
}) : () -> ()

