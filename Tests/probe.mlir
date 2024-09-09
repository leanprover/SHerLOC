"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "probe_i1"}> ({
    %29 = "stablehlo.constant"() <{value = dense<false> : tensor<3x1xi1>}> : () -> tensor<3x1xi1>
    %30 = "stablehlo.constant"() <{value = dense<true> : tensor<3x1xi1>}> : () -> tensor<3x1xi1>
    %31 = "stablehlo.add"(%29, %30) : (tensor<3x1xi1>, tensor<3x1xi1>) -> tensor<3x1xi1>
    %32 = "interpreter.probe"(%31) <{probe_id = "probe_i1"}> : (tensor<3x1xi1>) -> tensor<3x1xi1>
    "check.expect_serialized_eq"(%32) <{probe_id = "probe_i1"}> : (tensor<3x1xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "probe_si64"}> ({
    %25 = "stablehlo.constant"() <{value = dense<[[-127], [126], [0]]> : tensor<3x1xi64>}> : () -> tensor<3x1xi64>
    %26 = "stablehlo.constant"() <{value = dense<[[-1], [1], [1]]> : tensor<3x1xi64>}> : () -> tensor<3x1xi64>
    %27 = "stablehlo.add"(%25, %26) : (tensor<3x1xi64>, tensor<3x1xi64>) -> tensor<3x1xi64>
    %28 = "interpreter.probe"(%27) <{probe_id = "probe_si64"}> : (tensor<3x1xi64>) -> tensor<3x1xi64>
    "check.expect_serialized_eq"(%28) <{probe_id = "probe_si64"}> : (tensor<3x1xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "probe_ui64"}> ({
    %21 = "stablehlo.constant"() <{value = dense<[[32766, 0], [0, 0]]> : tensor<2x2xui64>}> : () -> tensor<2x2xui64>
    %22 = "stablehlo.constant"() <{value = dense<[[1, 1], [2, 2]]> : tensor<2x2xui64>}> : () -> tensor<2x2xui64>
    %23 = "stablehlo.add"(%21, %22) : (tensor<2x2xui64>, tensor<2x2xui64>) -> tensor<2x2xui64>
    %24 = "interpreter.probe"(%23) <{probe_id = "probe_ui64"}> : (tensor<2x2xui64>) -> tensor<2x2xui64>
    "check.expect_serialized_eq"(%24) <{probe_id = "probe_ui64"}> : (tensor<2x2xui64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "probe_f64"}> ({
    %17 = "stablehlo.constant"() <{value = dense<[3.402000e+38, 1.175000e-38, 0.000000e+00]> : tensor<3xf64>}> : () -> tensor<3xf64>
    %18 = "stablehlo.constant"() <{value = dense<[1.000000e+00, -1.000000e+00, 1.000000e+00]> : tensor<3xf64>}> : () -> tensor<3xf64>
    %19 = "stablehlo.add"(%17, %18) : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
    %20 = "interpreter.probe"(%19) <{probe_id = "probe_f64"}> : (tensor<3xf64>) -> tensor<3xf64>
    "check.expect_serialized_eq"(%20) <{probe_id = "probe_f64"}> : (tensor<3xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "probe_c32"}> ({
    %13 = "stablehlo.constant"() <{value = dense<[(1.000000e+00,2.000000e+00), (3.000000e+00,4.000000e+00), (5.000000e+00,6.000000e+00)]> : tensor<3xcomplex<f32>>}> : () -> tensor<3xcomplex<f32>>
    %14 = "stablehlo.constant"() <{value = dense<(1.000000e+00,1.000000e+00)> : tensor<3xcomplex<f32>>}> : () -> tensor<3xcomplex<f32>>
    %15 = "stablehlo.add"(%13, %14) : (tensor<3xcomplex<f32>>, tensor<3xcomplex<f32>>) -> tensor<3xcomplex<f32>>
    %16 = "interpreter.probe"(%15) <{probe_id = "probe_c32"}> : (tensor<3xcomplex<f32>>) -> tensor<3xcomplex<f32>>
    "check.expect_serialized_eq"(%16) <{probe_id = "probe_c32"}> : (tensor<3xcomplex<f32>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "probe_sanitized_probe_id"}> ({
    %9 = "stablehlo.constant"() <{value = dense<[[1], [2], [3]]> : tensor<3x1xi64>}> : () -> tensor<3x1xi64>
    %10 = "stablehlo.constant"() <{value = dense<[[4], [5], [6]]> : tensor<3x1xi64>}> : () -> tensor<3x1xi64>
    %11 = "stablehlo.add"(%9, %10) : (tensor<3x1xi64>, tensor<3x1xi64>) -> tensor<3x1xi64>
    %12 = "interpreter.probe"(%11) <{probe_id = "probe/0"}> : (tensor<3x1xi64>) -> tensor<3x1xi64>
    "check.expect_serialized_eq"(%12) <{probe_id = "probe/0"}> : (tensor<3x1xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "probe_iterations"}> ({
    %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %1 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %2 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %3 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
    %4:2 = "stablehlo.while"(%0, %1) ({
    ^bb0(%arg2: tensor<i64>, %arg3: tensor<i64>):
      %8 = "stablehlo.compare"(%arg2, %3) <{comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "stablehlo.return"(%8) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %5 = "stablehlo.add"(%arg1, %2) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      %6 = "interpreter.probe"(%5) <{probe_id = "probe_iterations"}> : (tensor<i64>) -> tensor<i64>
      %7 = "stablehlo.add"(%arg0, %2) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%7, %6) : (tensor<i64>, tensor<i64>) -> ()
    }) : (tensor<i64>, tensor<i64>) -> (tensor<i64>, tensor<i64>)
    "check.expect_eq_const"(%4#0) <{value = dense<2> : tensor<i64>}> : (tensor<i64>) -> ()
    "check.expect_serialized_eq"(%2) <{iteration = 0 : ui32, probe_id = "probe_iterations"}> : (tensor<i64>) -> ()
    "check.expect_serialized_eq"(%3) <{iteration = 1 : ui32, probe_id = "probe_iterations"}> : (tensor<i64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

