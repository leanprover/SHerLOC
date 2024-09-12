"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x4xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<3x4xcomplex<f64>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x4xcomplex<f64>>
    %4 = "stablehlo.real"(%2) : (tensor<3x4xcomplex<f64>>) -> tensor<3x4xf64>
    %5 = "stablehlo.imag"(%2) : (tensor<3x4xcomplex<f64>>) -> tensor<3x4xf64>
    %6 = "stablehlo.negate"(%5) : (tensor<3x4xf64>) -> tensor<3x4xf64>
    %7 = "stablehlo.complex"(%4, %6) : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3x4xcomplex<f64>>
    "stablehlo.custom_call"(%7, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x4xcomplex<f64>>, tensor<3x4xcomplex<f64>>) -> ()
    "func.return"(%7) : (tensor<3x4xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x4xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[(-2.7757113803979814,2.5267782869731414), (1.992888780677561,0.71865309190141702), (-0.7875001393193215,-2.7237443061422169), (1.9000941560104727,-2.4956603839421039)], [(1.450950019069186,8.1688934962764747), (5.5157213985318672,-5.2448518436290303), (0.66197619089797932,3.3153452667243641), (-1.606452313252019,-1.4369754941605484)], [(-4.6290449066973105,3.8151155866122046), (6.4014787745251205,-1.0610978321015645), (-1.8782692857676464,-2.0500741375956322), (0.014332066692161645,-1.1266138718597651)]]> : tensor<3x4xcomplex<f64>>}> : () -> tensor<3x4xcomplex<f64>>
    "func.return"(%1) : (tensor<3x4xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x4xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(-2.7757113803979814,-2.5267782869731414), (1.992888780677561,-0.71865309190141702), (-0.7875001393193215,2.7237443061422169), (1.9000941560104727,2.4956603839421039)], [(1.450950019069186,-8.1688934962764747), (5.5157213985318672,5.2448518436290303), (0.66197619089797932,-3.3153452667243641), (-1.606452313252019,1.4369754941605484)], [(-4.6290449066973105,-3.8151155866122046), (6.4014787745251205,1.0610978321015645), (-1.8782692857676464,2.0500741375956322), (0.014332066692161645,1.1266138718597651)]]> : tensor<3x4xcomplex<f64>>}> : () -> tensor<3x4xcomplex<f64>>
    "func.return"(%0) : (tensor<3x4xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

