"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x3xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %4:3 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x3xf32>, tensor<3x1xf32>, tensor<2xi64>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x3xf32>
    %6 = "stablehlo.slice"(%4#2) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<2xi64>) -> tensor<1xi64>
    %7 = "stablehlo.reshape"(%6) : (tensor<1xi64>) -> tensor<i64>
    %8 = "stablehlo.slice"(%4#2) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<2xi64>) -> tensor<1xi64>
    %9 = "stablehlo.reshape"(%8) : (tensor<1xi64>) -> tensor<i64>
    %10 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %11 = "stablehlo.compare"(%7, %10) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %12 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
    %13 = "stablehlo.add"(%7, %12) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %14 = "stablehlo.select"(%11, %13, %7) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
    %15 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %16 = "stablehlo.compare"(%9, %15) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %17 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
    %18 = "stablehlo.add"(%9, %17) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %19 = "stablehlo.select"(%16, %18, %9) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
    %20 = "stablehlo.dynamic_update_slice"(%4#0, %4#1, %14, %19) : (tensor<5x3xf32>, tensor<3x1xf32>, tensor<i64>, tensor<i64>) -> tensor<5x3xf32>
    "stablehlo.custom_call"(%20, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<5x3xf32>, tensor<5x3xf32>) -> ()
    "func.return"(%20) : (tensor<5x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x3xf32>, tensor<3x1xf32>, tensor<2xi64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[1.37886965, 2.52819586, -1.66359174], [-1.99620152, 1.19541264, 2.62602615], [-1.68388045, 2.40509582, 0.507822573], [-3.40378928, -1.23367071, -1.15847433], [1.35954201, 3.23837733, 0.490934104]]> : tensor<5x3xf32>}> : () -> tensor<5x3xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[-3.86802173], [-0.476641953], [0.915991306]]> : tensor<3x1xf32>}> : () -> tensor<3x1xf32>
    %3 = "stablehlo.constant"() <{value = dense<1> : tensor<2xi64>}> : () -> tensor<2xi64>
    "func.return"(%1, %2, %3) : (tensor<5x3xf32>, tensor<3x1xf32>, tensor<2xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1.37886965, 2.52819586, -1.66359174], [-1.99620152, -3.86802173, 2.62602615], [-1.68388045, -0.476641953, 0.507822573], [-3.40378928, 0.915991306, -1.15847433], [1.35954201, 3.23837733, 0.490934104]]> : tensor<5x3xf32>}> : () -> tensor<5x3xf32>
    "func.return"(%0) : (tensor<5x3xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

