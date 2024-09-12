"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xi32>, tensor<3x6xf64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xf64>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xi32>) -> tensor<4x3xf64>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xf64>) -> tensor<3x6xf64>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xf64>, tensor<3x6xf64>) -> tensor<4x6xf64>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xf64>, tensor<4x6xf64>) -> ()
    "func.return"(%7) : (tensor<4x6xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xi32>, tensor<3x6xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0, 2, -1], [-3, 0, 0], [0, 3, 6], [2, -1, 2]]> : tensor<4x3xi32>}> : () -> tensor<4x3xi32>
    %2 = "stablehlo.constant"() <{value = dense<[[3.4659555552847219, 4.2255337100745543, -5.4673025341106625, 3.4800095548462737, -4.012818706512439, 1.5126177330295851], [0.70199582847948072, 4.4209751754148696, -2.9100556724790274, -2.0857092183319894, -2.1443107051698429, -0.61941513355702638], [-4.6295346350155251, 1.090658511076384, 2.1762306249534613, -3.0100553769102412, 0.11623367309129488, -1.570001568863195]]> : tensor<3x6xf64>}> : () -> tensor<3x6xf64>
    "func.return"(%1, %2) : (tensor<4x3xi32>, tensor<3x6xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[6.0335262919744865, 7.751291839753355, -7.9963419699115157, -1.1613630597537377, -4.4048550834309808, 0.33117130174914222], [-10.397866665854165, -12.676601130223663, 16.401907602331988, -10.440028664538822, 12.038456119537317, -4.5378531990887554], [-25.671220324654708, 19.806876592702913, 4.3272167322836852, -24.317459916457416, -5.7355300769617585, -11.27825481385025], [-3.0291539879410871, 6.2114092668870065, -3.6720881458353754, 3.0256175742040554, -5.6488593616724447, 0.50464746188980669]]> : tensor<4x6xf64>}> : () -> tensor<4x6xf64>
    "func.return"(%0) : (tensor<4x6xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

