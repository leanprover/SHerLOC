"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<4x6xf64>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x5xf64>
    %4 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %5 = "stablehlo.reduce_window"(%2, %4) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %6 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      "stablehlo.return"(%6) : (tensor<f64>) -> ()
    }) : (tensor<4x6xf64>, tensor<f64>) -> tensor<3x5xf64>
    "stablehlo.custom_call"(%5, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x5xf64>, tensor<3x5xf64>) -> ()
    "func.return"(%5) : (tensor<3x5xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-1.476679218597688, 2.0431316424698656, 3.8494337987637257, -1.08363294655146, -1.4434161453902417, -0.53526242799487644], [1.4392491346452876, -0.26102123543740829, -2.1599746841105572, 0.62490016576673846, -6.2721891031333659, 3.0606977917334515], [1.1824749984152432, 0.91804707732091594, -2.943773753481238, -3.6168826507005365, -1.2016922257354874, -3.1268082252224421], [0.33629952166437987, -5.4224933195938965, -4.4068957909920128, -0.18843731564639468, -3.4955070170733098, -0.54614499375548842]]> : tensor<4x6xf64>}> : () -> tensor<4x6xf64>
    "func.return"(%1) : (tensor<4x6xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[2.0431316424698656, 3.8494337987637257, 3.8494337987637257, 1.000000e+00, 3.0606977917334515], [1.4392491346452876, 1.000000e+00, 1.000000e+00, 1.000000e+00, 3.0606977917334515], [1.1824749984152432, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]]> : tensor<3x5xf64>}> : () -> tensor<3x5xf64>
    "func.return"(%0) : (tensor<3x5xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

