"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x2xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x3xcomplex<f64>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x2xcomplex<f64>>
    %4 = "stablehlo.reshape"(%2) : (tensor<2x3xcomplex<f64>>) -> tensor<3x2xcomplex<f64>>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x2xcomplex<f64>>, tensor<3x2xcomplex<f64>>) -> ()
    "func.return"(%4) : (tensor<3x2xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[(-1.722938445328158,-6.1887849433992859), (-0.073958819640521034,-2.545936194133934), (0.22891570948245527,0.72224451118857946)], [(0.43591571123138967,2.6209821608255051), (4.5898309260273464,-0.17641004400846497), (2.358777963541467,1.7903291239492709)]]> : tensor<2x3xcomplex<f64>>}> : () -> tensor<2x3xcomplex<f64>>
    "func.return"(%1) : (tensor<2x3xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x2xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(-1.722938445328158,-6.1887849433992859), (-0.073958819640521034,-2.545936194133934)], [(0.22891570948245527,0.72224451118857946), (0.43591571123138967,2.6209821608255051)], [(4.5898309260273464,-0.17641004400846497), (2.358777963541467,1.7903291239492709)]]> : tensor<3x2xcomplex<f64>>}> : () -> tensor<3x2xcomplex<f64>>
    "func.return"(%0) : (tensor<3x2xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

