"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xbf16>, tensor<2x7xbf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xbf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%7) : (tensor<bf16>) -> ()
    }) : (tensor<5x6x7xbf16>, tensor<2x2xi64>, tensor<2x7xbf16>) -> tensor<5x6x7xbf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<5x6x7xbf16>, tensor<5x6x7xbf16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xbf16>, tensor<2x7xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xE63CFCBF163F01BF793FB03F283E77C0A140BDC0CB3FB43D8BC0E64029C0F3BF28BF96BF7EC0C5C0E43FA73FA0BFD93F09C051BF6DC0154085BF03C00740344087C0083FD73FEEBF5240DA40BBBEC140BABFBA3E16C060403940C8402340E83FE5403240DF3F8C402DC0BF3FA6BF18BF7F3F46C0F6BE9FBFB93E88404C4045BF62BF81C028C0E1BE6EBF223D3EBF71C02BBE57C02B40C13FF33D5D40853F00BF263F39C0EBC0D7BE1940994087BF953F8BC0AFBF0AC070C0154009C05940B73F44C0373F70C08C3FD5BF72C0F63F294063C00A3F8240D7BF49C050BE2140A040B0BFB4BFFA3E19BD65BF39403440B23F24400AC0F6404E3F73400DC0CC3FB23F0BBFD03FDF3E883F26409B3FA140FF3F583F8F3F3CC08940353FA9405ABFA04072C09E3F463FA5C0443F85C0124096C03FC01740AA3E9F3F93BF1D40E9BFE4BF5F3F9BC0C9BEE03E44C027C06FBF6340673E86C0514053BF903E8ABEC3BF55C0CE3F0EC08BBFCDBE2DC061BF43BFACBEAEBFD9BEC23F9F3D15C08D3E0BC093BF68C03940A23E3DBF293FA0BF03C0973D33C060C07FC0C5C04B4080BF97BF27C052C016C0"> : tensor<5x6x7xbf16>}> : () -> tensor<5x6x7xbf16>
    %2 = "stablehlo.constant"() <{value = dense<[[1.921880e+00, 4.375000e+00, -2.656250e-01, -5.351560e-01, 2.221680e-02, 2.578130e+00, 7.226560e-01], [1.093750e+00, -4.812500e+00, 7.500000e-01, -3.945310e-01, -7.890630e-01, -1.101560e+00, 1.984380e+00]]> : tensor<2x7xbf16>}> : () -> tensor<2x7xbf16>
    "func.return"(%1, %2) : (tensor<5x6x7xbf16>, tensor<2x7xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xE63CFCBF163F01BF793FB03F283EF63FA14088BECB3FB43D2540E64029C0F3BF28BF96BF7EC0C5C0E43FA73FA0BFD93F09C051BF6DC0154085BF03C00740344087C0083FD73FEEBF5240DA40BBBEC140BABFBA3E16C060403940C8402340E83FE5403240DF3F8C402DC0BF3FA6BF18BF7F3F46C0F6BE9FBFB93E88404C4045BF62BF81C028C0E1BE6EBF223D3EBF71C02BBE57C02B40C13FF33D5D40853F00BF263F39C0EBC0D7BE1940994087BF953F8BC0AFBF0AC070C0154009C05940B73F44C0373F70C08C3FD5BF72C0F63F294063C08C3F8240403FCABE50BE2140A040B0BFB4BFFA3E19BD65BF39403440B23F24400AC0F6404E3F73400DC0CC3FB23F0BBFD03FDF3E883F26409B3FA140FF3F583F8F3F3CC08940353FA9405ABFA04072C09E3F463FA5C0443F85C0124096C03FC01740AA3E9F3F93BF1D40E9BFE4BF5F3F9BC0C9BEE03E44C027C06FBF6340673E86C0514053BF903E8ABEC3BF55C0CE3F0EC08BBFCDBE2DC061BF43BFACBEAEBFD9BEC23F9F3D15C08D3E0BC093BF68C03940A23E3DBF293FA0BF03C0973D33C060C07FC0C5C04B4080BF97BF27C052C016C0"> : tensor<5x6x7xbf16>}> : () -> tensor<5x6x7xbf16>
    "func.return"(%0) : (tensor<5x6x7xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

