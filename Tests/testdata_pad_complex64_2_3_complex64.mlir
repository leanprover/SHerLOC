"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<6x4xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<6x4xcomplex<f32>>
    %5 = "stablehlo.pad"(%3#0, %3#1) <{edge_padding_high = array<i64: 2, 1>, edge_padding_low = array<i64: 1, 0>, interior_padding = array<i64: 1, 0>}> : (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<6x4xcomplex<f32>>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<6x4xcomplex<f32>>, tensor<6x4xcomplex<f32>>) -> ()
    "func.return"(%5) : (tensor<6x4xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[(-1.39428914E-4,8.64812405E-4), (-0.00152378506,-0.0014140791), (-0.00155372021,5.68573596E-4)], [(1.94506283E-4,0.00147564197), (-0.00122765638,-2.12210784E-4), (-3.98263692E-5,4.93056548E-4)]]> : tensor<2x3xcomplex<f32>>}> : () -> tensor<2x3xcomplex<f32>>
    %2 = "stablehlo.constant"() <{value = dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>}> : () -> tensor<complex<f32>>
    "func.return"(%1, %2) : (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<6x4xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)], [(-1.39428914E-4,8.64812405E-4), (-0.00152378506,-0.0014140791), (-0.00155372021,5.68573596E-4), (0.000000e+00,0.000000e+00)], [(0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)], [(1.94506283E-4,0.00147564197), (-0.00122765638,-2.12210784E-4), (-3.98263692E-5,4.93056548E-4), (0.000000e+00,0.000000e+00)], [(0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)], [(0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)]]> : tensor<6x4xcomplex<f32>>}> : () -> tensor<6x4xcomplex<f32>>
    "func.return"(%0) : (tensor<6x4xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

