"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x1xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x3xf32>, tensor<2xi64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2x1xf32>
    %5 = "stablehlo.slice"(%3#1) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<2xi64>) -> tensor<1xi64>
    %6 = "stablehlo.reshape"(%5) : (tensor<1xi64>) -> tensor<i64>
    %7 = "stablehlo.slice"(%3#1) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<2xi64>) -> tensor<1xi64>
    %8 = "stablehlo.reshape"(%7) : (tensor<1xi64>) -> tensor<i64>
    %9 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %10 = "stablehlo.compare"(%6, %9) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %11 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
    %12 = "stablehlo.add"(%6, %11) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %13 = "stablehlo.select"(%10, %12, %6) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
    %14 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %15 = "stablehlo.compare"(%8, %14) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %16 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
    %17 = "stablehlo.add"(%8, %16) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %18 = "stablehlo.select"(%15, %17, %8) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
    %19 = "stablehlo.dynamic_slice"(%3#0, %13, %18) <{slice_sizes = array<i64: 2, 1>}> : (tensor<5x3xf32>, tensor<i64>, tensor<i64>) -> tensor<2x1xf32>
    "stablehlo.custom_call"(%19, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x1xf32>, tensor<2x1xf32>) -> ()
    "func.return"(%19) : (tensor<2x1xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x3xf32>, tensor<2xi64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-3.56213808, -6.8185358, -1.571560e+00], [-1.97369742, -5.09079027, -0.138681203], [2.03843045, 3.21314955, -1.21502125], [-3.25714731, -0.58797425, 2.16309762], [-4.6540575, -0.637134611, -2.76275158]]> : tensor<5x3xf32>}> : () -> tensor<5x3xf32>
    %2 = "stablehlo.constant"() <{value = dense<1> : tensor<2xi64>}> : () -> tensor<2xi64>
    "func.return"(%1, %2) : (tensor<5x3xf32>, tensor<2xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x1xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-5.09079027], [3.21314955]]> : tensor<2x1xf32>}> : () -> tensor<2x1xf32>
    "func.return"(%0) : (tensor<2x1xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

