"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<ui64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %15 = "func.call"() <{callee = @inputs}> : () -> tensor<15xf32>
    %16 = "func.call"() <{callee = @expected}> : () -> tensor<ui64>
    %17 = "func.call"(%15) <{callee = @argmax}> : (tensor<15xf32>) -> tensor<ui64>
    "stablehlo.custom_call"(%17, %16) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<ui64>, tensor<ui64>) -> ()
    "func.return"(%17) : (tensor<ui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<15xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %14 = "stablehlo.constant"() <{value = dense<[-3.43464923, 7.15940142, 2.93741393, -0.346413374, 1.37187719, -1.83673358, 5.99954605, 2.46723032, -2.66364694, 0.517188668, -5.9878273, 6.03587484, 2.35249352, 1.32477033, 2.85481739]> : tensor<15xf32>}> : () -> tensor<15xf32>
    "func.return"(%14) : (tensor<15xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<ui64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %13 = "stablehlo.constant"() <{value = dense<1> : tensor<ui64>}> : () -> tensor<ui64>
    "func.return"(%13) : (tensor<ui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<15xf32>) -> tensor<ui64>, sym_name = "argmax", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<15xf32>):
    %0 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<15xui64>
    %1 = "stablehlo.constant"() <{value = dense<0xFF800000> : tensor<f32>}> : () -> tensor<f32>
    %2 = "stablehlo.constant"() <{value = dense<0> : tensor<ui64>}> : () -> tensor<ui64>
    %3:2 = "stablehlo.reduce"(%arg0, %0, %1, %2) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<ui64>, %arg3: tensor<f32>, %arg4: tensor<ui64>):
      %4 = "stablehlo.compare"(%arg1, %arg3) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %5 = "stablehlo.compare"(%arg1, %arg1) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %6 = "stablehlo.or"(%4, %5) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %7 = "stablehlo.compare"(%arg1, %arg3) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %8 = "stablehlo.compare"(%arg2, %arg4) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<ui64>, tensor<ui64>) -> tensor<i1>
      %9 = "stablehlo.and"(%7, %8) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %10 = "stablehlo.or"(%6, %9) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %11 = "stablehlo.select"(%6, %arg1, %arg3) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
      %12 = "stablehlo.select"(%10, %arg2, %arg4) : (tensor<i1>, tensor<ui64>, tensor<ui64>) -> tensor<ui64>
      "stablehlo.return"(%11, %12) : (tensor<f32>, tensor<ui64>) -> ()
    }) : (tensor<15xf32>, tensor<15xui64>, tensor<f32>, tensor<ui64>) -> (tensor<f32>, tensor<ui64>)
    "func.return"(%3#1) : (tensor<ui64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

