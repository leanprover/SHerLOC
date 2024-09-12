"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x1xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xcomplex<f64>>, tensor<complex<f64>>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2x1xcomplex<f64>>
    %5 = "stablehlo.pad"(%3#0, %3#1) <{edge_padding_high = array<i64: 0, -1>, edge_padding_low = array<i64: 0, -1>, interior_padding = array<i64: 0, 0>}> : (tensor<2x3xcomplex<f64>>, tensor<complex<f64>>) -> tensor<2x1xcomplex<f64>>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x1xcomplex<f64>>, tensor<2x1xcomplex<f64>>) -> ()
    "func.return"(%5) : (tensor<2x1xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xcomplex<f64>>, tensor<complex<f64>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[(3.0371844027451186E-4,8.634863150028557E-5), (-0.0010977968750361337,-0.0020786308515255291), (-4.906680339284398E-4,0.0015106559069639339)], [(-0.001370980718313348,-2.4014019974697736E-4), (-3.5628964552930848E-4,-6.542394855209258E-4), (0.0017045839820314776,-0.0032647636846900247)]]> : tensor<2x3xcomplex<f64>>}> : () -> tensor<2x3xcomplex<f64>>
    %2 = "stablehlo.constant"() <{value = dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>>}> : () -> tensor<complex<f64>>
    "func.return"(%1, %2) : (tensor<2x3xcomplex<f64>>, tensor<complex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x1xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(-0.0010977968750361337,-0.0020786308515255291)], [(-3.5628964552930848E-4,-6.542394855209258E-4)]]> : tensor<2x1xcomplex<f64>>}> : () -> tensor<2x1xcomplex<f64>>
    "func.return"(%0) : (tensor<2x1xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

