"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x3xcomplex<f32>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3xcomplex<f32>>
    %4 = "stablehlo.constant"() <{value = dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>}> : () -> tensor<complex<f32>>
    %5 = "stablehlo.reduce"(%2, %4) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>):
      %6 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<complex<f32>>, tensor<complex<f32>>) -> tensor<complex<f32>>
      "stablehlo.return"(%6) : (tensor<complex<f32>>) -> ()
    }) : (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<3xcomplex<f32>>
    "stablehlo.custom_call"(%5, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3xcomplex<f32>>, tensor<3xcomplex<f32>>) -> ()
    "func.return"(%5) : (tensor<3xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[(-1.21528089,-2.54824376), (2.66414642,-0.670537173), (-1.84409964,-2.86288095)], [(-1.61518693,2.00412917), (0.720621347,-0.149525791), (1.0502249,4.87241888)]]> : tensor<2x3xcomplex<f32>>}> : () -> tensor<2x3xcomplex<f32>>
    "func.return"(%1) : (tensor<2x3xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(7.06991577,1.68031025), (1.81957817,-8.815620e-01), (12.0124359,-11.9918947)]> : tensor<3xcomplex<f32>>}> : () -> tensor<3xcomplex<f32>>
    "func.return"(%0) : (tensor<3xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

