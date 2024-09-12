"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<i32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %15 = "func.call"() <{callee = @inputs}> : () -> tensor<15xf64>
    %16 = "func.call"() <{callee = @expected}> : () -> tensor<i32>
    %17 = "func.call"(%15) <{callee = @argmax}> : (tensor<15xf64>) -> tensor<i32>
    "stablehlo.custom_call"(%17, %16) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<i32>, tensor<i32>) -> ()
    "func.return"(%17) : (tensor<i32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<15xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %14 = "stablehlo.constant"() <{value = dense<[-1.885969853022754, -0.89396216218560309, 2.5137608126316411, 3.6901875442347194, 4.086138604994801, -0.98840977589852219, -4.9788625818119705, 2.4828224357985529, 2.8575335566531654, -4.2761857257592375, -1.1962313660114641, -1.0964159455293676, 4.012573928691503, -0.86455568784927106, 5.259781735323255]> : tensor<15xf64>}> : () -> tensor<15xf64>
    "func.return"(%14) : (tensor<15xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<i32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %13 = "stablehlo.constant"() <{value = dense<14> : tensor<i32>}> : () -> tensor<i32>
    "func.return"(%13) : (tensor<i32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<15xf64>) -> tensor<i32>, sym_name = "argmax", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<15xf64>):
    %0 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<15xi32>
    %1 = "stablehlo.constant"() <{value = dense<0xFFF0000000000000> : tensor<f64>}> : () -> tensor<f64>
    %2 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %3:2 = "stablehlo.reduce"(%arg0, %0, %1, %2) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg1: tensor<f64>, %arg2: tensor<i32>, %arg3: tensor<f64>, %arg4: tensor<i32>):
      %4 = "stablehlo.compare"(%arg1, %arg3) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %5 = "stablehlo.compare"(%arg1, %arg1) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %6 = "stablehlo.or"(%4, %5) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %7 = "stablehlo.compare"(%arg1, %arg3) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %8 = "stablehlo.compare"(%arg2, %arg4) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %9 = "stablehlo.and"(%7, %8) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %10 = "stablehlo.or"(%6, %9) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %11 = "stablehlo.select"(%6, %arg1, %arg3) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
      %12 = "stablehlo.select"(%10, %arg2, %arg4) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
      "stablehlo.return"(%11, %12) : (tensor<f64>, tensor<i32>) -> ()
    }) : (tensor<15xf64>, tensor<15xi32>, tensor<f64>, tensor<i32>) -> (tensor<f64>, tensor<i32>)
    "func.return"(%3#1) : (tensor<i32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

