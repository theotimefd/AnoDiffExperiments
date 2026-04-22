

import os
import glob

from utils import custom_transforms
from utils.utils import *
import monai.transforms
from monai import transforms
from monai.data import CacheDataset, DataLoader
from monai.utils import first
from torch.utils.data import Subset


def _subset_range(dataset, start, stop):
    # Keep splits as lightweight index views rather than materialized slices.
    return Subset(dataset, range(start, stop))


def _subset_first_half(dataset):
    split_idx = len(dataset) // 2
    return _subset_range(dataset, 0, split_idx)


def _subset_second_half(dataset):
    split_idx = len(dataset) // 2
    return _subset_range(dataset, split_idx, len(dataset))

class SOOP():
    def __init__(self, 
                 args,
                 batch_size=64,
                 num_workers=4,
                 groups_to_load=['large', 'medium', 'small'],
                 pin_memory=True):
        
        self.args = args
        self.root_dir = args.root_dir
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.groups_to_load = groups_to_load
        self.pin_memory = pin_memory
    

        masks_transforms = transforms.Compose(
            [
                transforms.LoadImage(),
                transforms.EnsureChannelFirst(),
                transforms.ResizeWithPadOrCrop(spatial_size=(args.image_size, args.image_size, args.image_size)),
                custom_transforms.SetBackgroundToZero()
            ]
        )

        # large, medium and small group split
        # large: >15cm3, medium: 5-15cm3, small: <5cm3
        large_group = ['sub-1010', 'sub-1015', 'sub-1025', 'sub-1032', 'sub-1035', 'sub-1043', 'sub-1045', 'sub-1096', 'sub-1150', 'sub-1156', 'sub-1164', 'sub-118', 'sub-1200', 'sub-1204', 'sub-1209', 'sub-1213', 'sub-1227', 'sub-1232', 'sub-1246', 'sub-1258', 'sub-127', 'sub-1280', 'sub-1282', 'sub-1283', 'sub-1301', 'sub-1309', 'sub-1323', 'sub-135', 'sub-1354', 'sub-1358', 'sub-1373', 'sub-1379', 'sub-1382', 'sub-1395', 'sub-1410', 'sub-1413', 'sub-1422', 'sub-1429', 'sub-1432', 'sub-1445', 'sub-1449', 'sub-1478', 'sub-1480', 'sub-1485', 'sub-1488', 'sub-1507', 'sub-1508', 'sub-1545', 'sub-1555', 'sub-1569', 'sub-1612', 'sub-1637', 'sub-1656', 'sub-1670', 'sub-1719', 'sub-1725', 'sub-173', 'sub-1736', 'sub-190', 'sub-198', 'sub-2', 'sub-241', 'sub-247', 'sub-262', 'sub-278', 'sub-284', 'sub-294', 'sub-3', 'sub-303', 'sub-321', 'sub-338', 'sub-342', 'sub-344', 'sub-346', 'sub-359', 'sub-360', 'sub-366', 'sub-370', 'sub-374', 'sub-398', 'sub-400', 'sub-401', 'sub-412', 'sub-422', 'sub-433', 'sub-443', 'sub-446', 'sub-447', 'sub-457', 'sub-463', 'sub-466', 'sub-477', 'sub-525', 'sub-53', 'sub-530', 'sub-543', 'sub-56', 'sub-572', 'sub-579', 'sub-631', 'sub-634', 'sub-651', 'sub-661', 'sub-692', 'sub-698', 'sub-721', 'sub-723', 'sub-724', 'sub-751', 'sub-754', 'sub-760', 'sub-761', 'sub-768', 'sub-776', 'sub-789', 'sub-791', 'sub-796', 'sub-8', 'sub-803', 'sub-813', 'sub-826', 'sub-845', 'sub-858', 'sub-866', 'sub-873', 'sub-877', 'sub-896', 'sub-937', 'sub-942', 'sub-946', 'sub-959', 'sub-960', 'sub-990']
        medium_group = ['sub-100', 'sub-1006', 'sub-1011', 'sub-1014', 'sub-1016', 'sub-1018', 'sub-102', 'sub-1024', 'sub-103', 'sub-1052', 'sub-1054', 'sub-1055', 'sub-1057', 'sub-106', 'sub-1064', 'sub-1075', 'sub-1076', 'sub-1081', 'sub-1093', 'sub-1101', 'sub-1105', 'sub-1106', 'sub-1118', 'sub-1120', 'sub-1127', 'sub-1130', 'sub-1140', 'sub-1144', 'sub-1154', 'sub-1157', 'sub-1163', 'sub-1186', 'sub-1193', 'sub-1211', 'sub-1212', 'sub-1217', 'sub-122', 'sub-123', 'sub-1234', 'sub-1239', 'sub-124', 'sub-1244', 'sub-1248', 'sub-1260', 'sub-1266', 'sub-1277', 'sub-128', 'sub-1281', 'sub-1296', 'sub-1297', 'sub-1298', 'sub-1302', 'sub-1310', 'sub-1317', 'sub-1319', 'sub-1324', 'sub-1326', 'sub-1331', 'sub-1332', 'sub-1338', 'sub-1346', 'sub-1347', 'sub-1349', 'sub-1352', 'sub-1363', 'sub-1374', 'sub-1388', 'sub-1404', 'sub-1408', 'sub-1415', 'sub-1417', 'sub-1423', 'sub-1427', 'sub-1438', 'sub-1443', 'sub-1446', 'sub-145', 'sub-146', 'sub-1460', 'sub-1463', 'sub-147', 'sub-1489', 'sub-1494', 'sub-1501', 'sub-1503', 'sub-1509', 'sub-1518', 'sub-1519', 'sub-1521', 'sub-1522', 'sub-1523', 'sub-1541', 'sub-1547', 'sub-1548', 'sub-155', 'sub-1550', 'sub-1556', 'sub-1557', 'sub-156', 'sub-1565', 'sub-1578', 'sub-1595', 'sub-16', 'sub-1603', 'sub-1605', 'sub-1608', 'sub-161', 'sub-162', 'sub-1629', 'sub-1636', 'sub-1638', 'sub-1646', 'sub-165', 'sub-1652', 'sub-1672', 'sub-1673', 'sub-1674', 'sub-1678', 'sub-1682', 'sub-1684', 'sub-1690', 'sub-1722', 'sub-203', 'sub-204', 'sub-206', 'sub-219', 'sub-245', 'sub-27', 'sub-273', 'sub-274', 'sub-277', 'sub-289', 'sub-295', 'sub-296', 'sub-297', 'sub-305', 'sub-320', 'sub-322', 'sub-33', 'sub-330', 'sub-332', 'sub-35', 'sub-355', 'sub-36', 'sub-364', 'sub-375', 'sub-377', 'sub-384', 'sub-397', 'sub-403', 'sub-408', 'sub-415', 'sub-416', 'sub-420', 'sub-426', 'sub-435', 'sub-449', 'sub-462', 'sub-467', 'sub-473', 'sub-478', 'sub-485', 'sub-487', 'sub-49', 'sub-490', 'sub-50', 'sub-507', 'sub-510', 'sub-515', 'sub-518', 'sub-522', 'sub-542', 'sub-544', 'sub-546', 'sub-551', 'sub-552', 'sub-554', 'sub-557', 'sub-580', 'sub-587', 'sub-589', 'sub-595', 'sub-596', 'sub-616', 'sub-62', 'sub-622', 'sub-654', 'sub-657', 'sub-663', 'sub-67', 'sub-674', 'sub-681', 'sub-703', 'sub-717', 'sub-759', 'sub-794', 'sub-795', 'sub-801', 'sub-807', 'sub-821', 'sub-839', 'sub-848', 'sub-853', 'sub-860', 'sub-869', 'sub-870', 'sub-880', 'sub-888', 'sub-9', 'sub-908', 'sub-918', 'sub-924', 'sub-927', 'sub-931', 'sub-943', 'sub-944', 'sub-947', 'sub-965', 'sub-969', 'sub-972', 'sub-976', 'sub-991', 'sub-996', 'sub-998']
        small_group = ['sub-10', 'sub-1000', 'sub-1001', 'sub-1003', 'sub-1004', 'sub-1005', 'sub-1007', 'sub-1009', 'sub-101', 'sub-1019', 'sub-1020', 'sub-1021', 'sub-1026', 'sub-1027', 'sub-1028', 'sub-1029', 'sub-1036', 'sub-1037', 'sub-1038', 'sub-104', 'sub-1040', 'sub-1042', 'sub-1047', 'sub-1049', 'sub-105', 'sub-1050', 'sub-1058', 'sub-1062', 'sub-1063', 'sub-1066', 'sub-1067', 'sub-1068', 'sub-1069', 'sub-1072', 'sub-1077', 'sub-108', 'sub-1083', 'sub-1084', 'sub-1085', 'sub-1087', 'sub-1089', 'sub-1092', 'sub-1099', 'sub-1100', 'sub-1104', 'sub-1108', 'sub-1109', 'sub-111', 'sub-1110', 'sub-1117', 'sub-1121', 'sub-1122', 'sub-1124', 'sub-1129', 'sub-1132', 'sub-1134', 'sub-1137', 'sub-1143', 'sub-1145', 'sub-115', 'sub-1152', 'sub-1153', 'sub-1155', 'sub-1159', 'sub-1162', 'sub-1168', 'sub-117', 'sub-1171', 'sub-1177', 'sub-1184', 'sub-1185', 'sub-1187', 'sub-1188', 'sub-119', 'sub-1190', 'sub-1191', 'sub-1195', 'sub-1197', 'sub-1199', 'sub-1201', 'sub-1206', 'sub-1207', 'sub-1208', 'sub-121', 'sub-1210', 'sub-1214', 'sub-1216', 'sub-1219', 'sub-1221', 'sub-1222', 'sub-1224', 'sub-1226', 'sub-1228', 'sub-1230', 'sub-1233', 'sub-1236', 'sub-1238', 'sub-1240', 'sub-1243', 'sub-1245', 'sub-1247', 'sub-1249', 'sub-1250', 'sub-1252', 'sub-1253', 'sub-1257', 'sub-1259', 'sub-126', 'sub-1263', 'sub-1265', 'sub-1268', 'sub-1270', 'sub-1271', 'sub-1272', 'sub-1273', 'sub-1274', 'sub-1275', 'sub-1278', 'sub-1279', 'sub-1286', 'sub-1287', 'sub-1288', 'sub-1294', 'sub-1295', 'sub-1299', 'sub-1300', 'sub-1303', 'sub-1304', 'sub-1315', 'sub-1316', 'sub-132', 'sub-1321', 'sub-1325', 'sub-1327', 'sub-1328', 'sub-1333', 'sub-1336', 'sub-1341', 'sub-1343', 'sub-1350', 'sub-1351', 'sub-1353', 'sub-1356', 'sub-1359', 'sub-136', 'sub-1362', 'sub-1367', 'sub-137', 'sub-1375', 'sub-1376', 'sub-1377', 'sub-1378', 'sub-1383', 'sub-1387', 'sub-139', 'sub-1391', 'sub-1392', 'sub-1393', 'sub-1394', 'sub-1397', 'sub-1398', 'sub-1399', 'sub-14', 'sub-1401', 'sub-1405', 'sub-1411', 'sub-1412', 'sub-1414', 'sub-1416', 'sub-142', 'sub-1420', 'sub-1425', 'sub-1426', 'sub-1430', 'sub-1431', 'sub-1435', 'sub-1439', 'sub-1441', 'sub-1451', 'sub-1457', 'sub-1464', 'sub-1465', 'sub-1467', 'sub-1469', 'sub-1471', 'sub-1472', 'sub-1473', 'sub-1474', 'sub-1476', 'sub-1477', 'sub-1479', 'sub-1481', 'sub-1482', 'sub-1484', 'sub-1486', 'sub-1487', 'sub-149', 'sub-1493', 'sub-1497', 'sub-1498', 'sub-1499', 'sub-15', 'sub-150', 'sub-1505', 'sub-1510', 'sub-1515', 'sub-1516', 'sub-152', 'sub-1524', 'sub-1527', 'sub-1528', 'sub-1530', 'sub-1531', 'sub-1535', 'sub-1537', 'sub-1538', 'sub-1542', 'sub-1543', 'sub-1546', 'sub-1549', 'sub-1559', 'sub-1560', 'sub-1561', 'sub-157', 'sub-1572', 'sub-1574', 'sub-1576', 'sub-1577', 'sub-1579', 'sub-158', 'sub-1580', 'sub-1581', 'sub-1582', 'sub-1584', 'sub-1585', 'sub-1586', 'sub-1587', 'sub-1593', 'sub-1594', 'sub-1596', 'sub-1597', 'sub-1599', 'sub-1600', 'sub-1601', 'sub-1606', 'sub-1614', 'sub-1615', 'sub-1618', 'sub-1619', 'sub-1620', 'sub-1621', 'sub-1622', 'sub-1625', 'sub-1627', 'sub-1628', 'sub-163', 'sub-1632', 'sub-1633', 'sub-1635', 'sub-1639', 'sub-164', 'sub-1642', 'sub-1643', 'sub-1645', 'sub-1650', 'sub-1653', 'sub-1655', 'sub-166', 'sub-1661', 'sub-1662', 'sub-1663', 'sub-1664', 'sub-1666', 'sub-1668', 'sub-1669', 'sub-1671', 'sub-1676', 'sub-1679', 'sub-168', 'sub-1680', 'sub-1685', 'sub-1686', 'sub-1687', 'sub-169', 'sub-1691', 'sub-1692', 'sub-1693', 'sub-1694', 'sub-1699', 'sub-17', 'sub-170', 'sub-1700', 'sub-1704', 'sub-1705', 'sub-1706', 'sub-1709', 'sub-1711', 'sub-1712', 'sub-1714', 'sub-1716', 'sub-172', 'sub-1720', 'sub-1721', 'sub-1723', 'sub-1726', 'sub-1728', 'sub-1729', 'sub-1734', 'sub-1735', 'sub-1737', 'sub-176', 'sub-178', 'sub-180', 'sub-181', 'sub-182', 'sub-183', 'sub-186', 'sub-189', 'sub-19', 'sub-192', 'sub-194', 'sub-195', 'sub-197', 'sub-200', 'sub-201', 'sub-202', 'sub-207', 'sub-21', 'sub-211', 'sub-212', 'sub-213', 'sub-214', 'sub-216', 'sub-217', 'sub-218', 'sub-220', 'sub-222', 'sub-223', 'sub-226', 'sub-229', 'sub-23', 'sub-230', 'sub-239', 'sub-24', 'sub-240', 'sub-242', 'sub-246', 'sub-248', 'sub-250', 'sub-252', 'sub-253', 'sub-254', 'sub-255', 'sub-257', 'sub-258', 'sub-259', 'sub-26', 'sub-261', 'sub-263', 'sub-265', 'sub-266', 'sub-267', 'sub-268', 'sub-270', 'sub-275', 'sub-276', 'sub-280', 'sub-281', 'sub-285', 'sub-287', 'sub-288', 'sub-290', 'sub-291', 'sub-292', 'sub-293', 'sub-298', 'sub-299', 'sub-30', 'sub-300', 'sub-306', 'sub-309', 'sub-310', 'sub-311', 'sub-312', 'sub-313', 'sub-315', 'sub-318', 'sub-324', 'sub-325', 'sub-327', 'sub-336', 'sub-340', 'sub-347', 'sub-349', 'sub-350', 'sub-351', 'sub-354', 'sub-356', 'sub-357', 'sub-362', 'sub-365', 'sub-368', 'sub-373', 'sub-38', 'sub-385', 'sub-390', 'sub-392', 'sub-393', 'sub-394', 'sub-395', 'sub-396', 'sub-399', 'sub-4', 'sub-405', 'sub-406', 'sub-407', 'sub-41', 'sub-410', 'sub-411', 'sub-418', 'sub-419', 'sub-423', 'sub-424', 'sub-425', 'sub-427', 'sub-429', 'sub-43', 'sub-430', 'sub-431', 'sub-437', 'sub-439', 'sub-44', 'sub-440', 'sub-442', 'sub-445', 'sub-45', 'sub-450', 'sub-452', 'sub-456', 'sub-458', 'sub-46', 'sub-461', 'sub-468', 'sub-469', 'sub-472', 'sub-474', 'sub-475', 'sub-479', 'sub-480', 'sub-481', 'sub-484', 'sub-486', 'sub-489', 'sub-492', 'sub-493', 'sub-496', 'sub-5', 'sub-502', 'sub-503', 'sub-504', 'sub-514', 'sub-516', 'sub-520', 'sub-527', 'sub-528', 'sub-531', 'sub-532', 'sub-533', 'sub-534', 'sub-536', 'sub-540', 'sub-549', 'sub-556', 'sub-558', 'sub-559', 'sub-562', 'sub-564', 'sub-565', 'sub-566', 'sub-567', 'sub-569', 'sub-573', 'sub-577', 'sub-58', 'sub-581', 'sub-583', 'sub-584', 'sub-585', 'sub-59', 'sub-590', 'sub-591', 'sub-593', 'sub-597', 'sub-598', 'sub-599', 'sub-60', 'sub-603', 'sub-604', 'sub-606', 'sub-607', 'sub-609', 'sub-615', 'sub-618', 'sub-619', 'sub-624', 'sub-628', 'sub-63', 'sub-630', 'sub-632', 'sub-636', 'sub-637', 'sub-639', 'sub-64', 'sub-643', 'sub-645', 'sub-649', 'sub-650', 'sub-653', 'sub-655', 'sub-656', 'sub-658', 'sub-659', 'sub-66', 'sub-660', 'sub-662', 'sub-665', 'sub-675', 'sub-676', 'sub-677', 'sub-687', 'sub-688', 'sub-690', 'sub-693', 'sub-696', 'sub-697', 'sub-700', 'sub-701', 'sub-702', 'sub-704', 'sub-705', 'sub-708', 'sub-709', 'sub-71', 'sub-712', 'sub-714', 'sub-715', 'sub-716', 'sub-718', 'sub-725', 'sub-727', 'sub-73', 'sub-730', 'sub-731', 'sub-732', 'sub-733', 'sub-734', 'sub-744', 'sub-745', 'sub-748', 'sub-749', 'sub-750', 'sub-753', 'sub-755', 'sub-758', 'sub-762', 'sub-763', 'sub-769', 'sub-77', 'sub-773', 'sub-774', 'sub-775', 'sub-777', 'sub-778', 'sub-780', 'sub-783', 'sub-784', 'sub-786', 'sub-787', 'sub-790', 'sub-792', 'sub-799', 'sub-80', 'sub-800', 'sub-805', 'sub-809', 'sub-81', 'sub-810', 'sub-812', 'sub-814', 'sub-815', 'sub-816', 'sub-819', 'sub-820', 'sub-825', 'sub-827', 'sub-828', 'sub-83', 'sub-832', 'sub-833', 'sub-835', 'sub-838', 'sub-84', 'sub-840', 'sub-841', 'sub-849', 'sub-850', 'sub-852', 'sub-854', 'sub-857', 'sub-86', 'sub-863', 'sub-868', 'sub-87', 'sub-872', 'sub-875', 'sub-876', 'sub-884', 'sub-886', 'sub-89', 'sub-891', 'sub-892', 'sub-893', 'sub-895', 'sub-898', 'sub-90', 'sub-900', 'sub-903', 'sub-905', 'sub-906', 'sub-907', 'sub-91', 'sub-913', 'sub-914', 'sub-915', 'sub-919', 'sub-92', 'sub-920', 'sub-921', 'sub-923', 'sub-925', 'sub-926', 'sub-928', 'sub-929', 'sub-930', 'sub-932', 'sub-934', 'sub-935', 'sub-94', 'sub-941', 'sub-945', 'sub-948', 'sub-951', 'sub-955', 'sub-956', 'sub-957', 'sub-958', 'sub-961', 'sub-97', 'sub-970', 'sub-971', 'sub-973', 'sub-974', 'sub-975', 'sub-978', 'sub-979', 'sub-98', 'sub-983', 'sub-986', 'sub-988', 'sub-992', 'sub-993', 'sub-994', 'sub-995']

        if "flair" in self.args.dataset["name"].lower():
            test_anomaly_images = sorted(glob.glob(self.root_dir+"datasets/final_soop_dataset_small/flair_registered/*.nii.gz"))
        elif "adc" in self.args.dataset["name"].lower():
            test_anomaly_images = sorted(glob.glob(self.root_dir+"datasets/final_soop_dataset_small/adc_registered/*.nii.gz"))

        tests_anomaly_masks = glob.glob(self.root_dir+"datasets/final_soop_dataset_small/masks_combined_registered/*.nii.gz")


        images_to_exclude = []
        with open(self.root_dir+"AnoDiffExperiments/data_splits_lists/final_soop_dataset_small/exclude.csv", 'r') as f:
            for line in f:
                images_to_exclude.append(line.strip())

        with open(self.root_dir+"AnoDiffExperiments/data_splits_lists/final_soop_dataset_small/exclude_non_axial_thick_slices.csv", 'r') as f:
            for line in f:
                images_to_exclude.append(line.strip())
        
        test_anomaly_transforms = define_instance(self.args, "val_transforms")

        # final lists of images and masks path after excluding the bad images
        self.test_anomaly_large_images = [path for path in test_anomaly_images if os.path.basename(path).split('.')[0] not in images_to_exclude and os.path.basename(path).split('.')[0] in large_group]        
        self.large_group_masks = [path for path in tests_anomaly_masks if os.path.basename(path).split('.')[0] not in images_to_exclude and os.path.basename(path).split('.')[0] in large_group]

        self.test_anomaly_medium_images = [path for path in test_anomaly_images if os.path.basename(path).split('.')[0] not in images_to_exclude and os.path.basename(path).split('.')[0] in medium_group]        
        self.medium_group_masks = [path for path in tests_anomaly_masks if os.path.basename(path).split('.')[0] not in images_to_exclude and os.path.basename(path).split('.')[0] in medium_group]

        self.test_anomaly_small_images = [path for path in test_anomaly_images if os.path.basename(path).split('.')[0] not in images_to_exclude and os.path.basename(path).split('.')[0] in small_group]        
        self.small_group_masks = [path for path in tests_anomaly_masks if os.path.basename(path).split('.')[0] not in images_to_exclude and os.path.basename(path).split('.')[0] in small_group]
        
        self.test_anomaly_large_images = sorted(self.test_anomaly_large_images, key=lambda x: os.path.basename(x).split('.')[0])
        self.large_group_masks = sorted(self.large_group_masks, key=lambda x: os.path.basename(x).split('.')[0])

        self.test_anomaly_medium_images = sorted(self.test_anomaly_medium_images, key=lambda x: os.path.basename(x).split('.')[0])
        self.medium_group_masks = sorted(self.medium_group_masks, key=lambda x: os.path.basename(x).split('.')[0])

        self.test_anomaly_small_images = sorted(self.test_anomaly_small_images, key=lambda x: os.path.basename(x).split('.')[0])
        self.small_group_masks = sorted(self.small_group_masks, key=lambda x: os.path.basename(x).split('.')[0])

        self.test_anomaly_small_images = self.test_anomaly_small_images[:200] # Test set : SOOP: we only kept the first 200 small group images otherwis takes too much time
        self.small_group_masks = self.small_group_masks[:200] # Test set : SOOP: we only kept the first 200 small group images otherwis takes too much time



        # dataloaders
        # each group of images is split into two halves:
        # select params half: used to select the best noise timestep value, best threshold etc
        # metrics half: used to compute the final scores (e.g DICE) with these best values.

        
        if "large" in self.groups_to_load:
            
            test_anomaly_large_ds = CacheDataset(data=self.test_anomaly_large_images, transform=test_anomaly_transforms)
            self.test_anomaly_large_loader_select_params = DataLoader( 
                _subset_first_half(test_anomaly_large_ds), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory
            )
            self.test_anomaly_large_images_select_params = self.test_anomaly_large_images[:len(self.test_anomaly_large_images)//2]

            self.test_anomaly_large_loader_metrics = DataLoader(       
                _subset_second_half(test_anomaly_large_ds), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory
            )
            self.test_anomaly_large_images_metrics = self.test_anomaly_large_images[len(self.test_anomaly_large_images)//2:]

            ## masks
            self.test_masks_large_ds = CacheDataset(data=self.large_group_masks, transform=masks_transforms)
            self.test_masks_large_loader_select_params = DataLoader(
                _subset_first_half(self.test_masks_large_ds), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory
            )
            self.test_masks_large_loader_metrics = DataLoader(
                _subset_second_half(self.test_masks_large_ds), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory
            )

        if "medium" in self.groups_to_load:
            test_anomaly_medium_ds = CacheDataset(data=self.test_anomaly_medium_images, transform=test_anomaly_transforms)
            self.test_anomaly_medium_loader_select_params = DataLoader(
                _subset_first_half(test_anomaly_medium_ds), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory
            )
            self.test_anomaly_medium_images_select_params = self.test_anomaly_medium_images[:len(self.test_anomaly_medium_images)//2]

            self.test_anomaly_medium_loader_metrics = DataLoader(
                _subset_second_half(test_anomaly_medium_ds), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory
            )
            self.test_anomaly_medium_images_metrics = self.test_anomaly_medium_images[len(self.test_anomaly_medium_images)//2:]

            ## masks
            self.test_masks_medium_ds = CacheDataset(data=self.medium_group_masks, transform=masks_transforms)
            self.test_masks_medium_loader_select_params = DataLoader(
                _subset_first_half(self.test_masks_medium_ds), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory
            )
            self.test_masks_medium_loader_metrics = DataLoader(
                _subset_second_half(self.test_masks_medium_ds), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory
            )

        if "small" in self.groups_to_load:
            # small group
            test_anomaly_small_ds = CacheDataset(data=self.test_anomaly_small_images, transform=test_anomaly_transforms)
            self.test_anomaly_small_loader_select_params = DataLoader(
                _subset_first_half(test_anomaly_small_ds), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory
            )
            self.test_anomaly_small_images_select_params = self.test_anomaly_small_images[:len(self.test_anomaly_small_images)//2]

            self.test_anomaly_small_loader_metrics = DataLoader(
                _subset_second_half(test_anomaly_small_ds), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory
            )
            self.test_anomaly_small_images_metrics = self.test_anomaly_small_images[len(self.test_anomaly_small_images)//2:]
            
            ## masks
            self.test_masks_small_ds = CacheDataset(data=self.small_group_masks, transform=masks_transforms)
            self.test_masks_small_loader_select_params = DataLoader(
                _subset_first_half(self.test_masks_small_ds), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory
            )
            self.test_masks_small_loader_metrics = DataLoader(
                _subset_second_half(self.test_masks_small_ds), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory
            )



    def len_large_group(self):
        if "large" not in self.groups_to_load:
            dtprint("Large group not loaded. Returning 0.")
            return 0
        return len(self.test_anomaly_large_images)
    
    def len_medium_group(self):
        if "medium" not in self.groups_to_load:
            dtprint("Medium group not loaded. Returning 0.")
            return 0
        return len(self.test_anomaly_medium_images)
    
    def len_small_group(self):
        if "small" not in self.groups_to_load:
            dtprint("Small group not loaded. Returning 0.")
            return 0
        return len(self.test_anomaly_small_images)

    def first(self):
        if "large" in self.groups_to_load:
            dtprint("Returning first sample from large group.")
            return first(self.test_anomaly_large_loader_select_params)
        elif "medium" in self.groups_to_load:
            dtprint("Returning first sample from medium group.")
            return first(self.test_anomaly_medium_loader_select_params)
        elif "small" in self.groups_to_load:
            dtprint("Returning first sample from small group.")
            return first(self.test_anomaly_small_loader_select_params)
    
    def get_anomaly_loader_select_params(self, group):
        if group == "large":
            return self.test_anomaly_large_loader_select_params
        elif group == "medium":
            return self.test_anomaly_medium_loader_select_params
        elif group == "small":
            return self.test_anomaly_small_loader_select_params
        else:
            raise ValueError("Invalid group name. Must be 'large', 'medium', or 'small'.")
    
    def get_anomaly_loader_metrics(self, group):
        if group == "large":
            return self.test_anomaly_large_loader_metrics
        elif group == "medium":
            return self.test_anomaly_medium_loader_metrics
        elif group == "small":
            return self.test_anomaly_small_loader_metrics
        else:
            raise ValueError("Invalid group name. Must be 'large', 'medium', or 'small'.")
    
    def get_masks_loader_select_params(self, group):
        if group == "large":
            return self.test_masks_large_loader_select_params
        elif group == "medium":
            return self.test_masks_medium_loader_select_params
        elif group == "small":
            return self.test_masks_small_loader_select_params
        else:
            raise ValueError("Invalid group name. Must be 'large', 'medium', or 'small'.")
    
    def get_masks_loader_metrics(self, group):
        if group == "large":
            return self.test_masks_large_loader_metrics
        elif group == "medium":
            return self.test_masks_medium_loader_metrics
        elif group == "small":
            return self.test_masks_small_loader_metrics
        else:
            raise ValueError("Invalid group name. Must be 'large', 'medium', or 'small'.")

    def get_anomaly_images_select_params(self, group):
        if group == "large":
            return self.test_anomaly_large_images_select_params
        elif group == "medium":
            return self.test_anomaly_medium_images_select_params
        elif group == "small":
            return self.test_anomaly_small_images_select_params
        else:
            raise ValueError("Invalid group name. Must be 'large', 'medium', or 'small'.")
    
    def get_anomaly_images_metrics(self, group):
        if group == "large":
            return self.test_anomaly_large_images_metrics
        elif group == "medium":
            return self.test_anomaly_medium_images_metrics
        elif group == "small":
            return self.test_anomaly_small_images_metrics
        else:
            raise ValueError("Invalid group name. Must be 'large', 'medium', or 'small'.")


class SOOP_large_only():
    def __init__(self, 
                 args,
                 batch_size=64,
                 num_workers=4,
                 num_images_to_load=-1,
                ):
        
        self.args = args
        self.root_dir = args.root_dir
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
    

        masks_transforms = transforms.Compose(
            [
                transforms.LoadImage(),
                transforms.EnsureChannelFirst(),
                transforms.ResizeWithPadOrCrop(spatial_size=(args.image_size, args.image_size, args.image_size)),
                custom_transforms.SetBackgroundToZero()
            ]
        )

        # large, medium and small group split
        # large: >15cm3, medium: 5-15cm3, small: <5cm3
        large_group = ['sub-1010', 'sub-1013', 'sub-1015', 'sub-1032', 'sub-1035', 'sub-1039', 'sub-1041', 'sub-1045', 'sub-1046', 'sub-1071', 'sub-1073', 'sub-1086', 'sub-1102', 'sub-1107', 'sub-1115', 'sub-113', 'sub-114', 'sub-1149', 'sub-1150', 'sub-116', 'sub-1164', 'sub-1165', 'sub-118', 'sub-1200', 'sub-1204', 'sub-1209', 'sub-1213', 'sub-1215', 'sub-1223', 'sub-1227', 'sub-1232', 'sub-1246', 'sub-1258', 'sub-127', 'sub-1280', 'sub-1282', 'sub-1283', 'sub-1285', 'sub-1292', 'sub-1305', 'sub-1306', 'sub-1309', 'sub-1312', 'sub-1314', 'sub-1320', 'sub-1323', 'sub-135', 'sub-1354', 'sub-1355', 'sub-1358', 'sub-1364', 'sub-1366', 'sub-1369', 'sub-1373', 'sub-1379', 'sub-1382', 'sub-1386', 'sub-1395', 'sub-1409', 'sub-1410', 'sub-1413', 'sub-1422', 'sub-1432', 'sub-1445', 'sub-1447', 'sub-1475', 'sub-1478', 'sub-1480', 'sub-1483', 'sub-1485', 'sub-1488', 'sub-1507', 'sub-1508', 'sub-1511', 'sub-1517', 'sub-1552', 'sub-1554', 'sub-1555', 'sub-1569', 'sub-1598', 'sub-1612', 'sub-1634', 'sub-1637', 'sub-1656', 'sub-1670', 'sub-1677', 'sub-1719', 'sub-1725', 'sub-1727', 'sub-1736', 'sub-174', 'sub-177', 'sub-185', 'sub-190', 'sub-196', 'sub-198', 'sub-2', 'sub-221', 'sub-235', 'sub-241', 'sub-247', 'sub-249', 'sub-260', 'sub-262', 'sub-264', 'sub-278', 'sub-284', 'sub-294', 'sub-3', 'sub-303', 'sub-314', 'sub-321', 'sub-326', 'sub-335', 'sub-338', 'sub-339', 'sub-341', 'sub-343', 'sub-345', 'sub-359', 'sub-360', 'sub-366', 'sub-370', 'sub-374', 'sub-386', 'sub-398', 'sub-400', 'sub-401', 'sub-412', 'sub-42', 'sub-422', 'sub-432', 'sub-433', 'sub-443', 'sub-446', 'sub-447', 'sub-457', 'sub-463', 'sub-464', 'sub-466', 'sub-47', 'sub-494', 'sub-498', 'sub-501', 'sub-505', 'sub-512', 'sub-517', 'sub-521', 'sub-523', 'sub-525', 'sub-529', 'sub-53', 'sub-530', 'sub-539', 'sub-543', 'sub-56', 'sub-563', 'sub-572', 'sub-613', 'sub-620', 'sub-631', 'sub-634', 'sub-638', 'sub-651', 'sub-652', 'sub-661', 'sub-682', 'sub-692', 'sub-694', 'sub-698', 'sub-699', 'sub-707', 'sub-719', 'sub-723', 'sub-724', 'sub-751', 'sub-754', 'sub-760', 'sub-761', 'sub-768', 'sub-776', 'sub-789', 'sub-79', 'sub-791', 'sub-8', 'sub-803', 'sub-806', 'sub-82', 'sub-823', 'sub-826', 'sub-843', 'sub-844', 'sub-845', 'sub-858', 'sub-861', 'sub-865', 'sub-866', 'sub-873', 'sub-877', 'sub-881', 'sub-896', 'sub-917', 'sub-937', 'sub-939', 'sub-942', 'sub-946', 'sub-95', 'sub-952', 'sub-959', 'sub-960', 'sub-968', 'sub-990']
        
        if "flair" in self.args.dataset["name"].lower():
            test_anomaly_images = sorted(glob.glob(self.root_dir+"datasets/final_soop_dataset_small/flair_registered/*.nii.gz"))
        elif "adc" in self.args.dataset["name"].lower():
            test_anomaly_images = sorted(glob.glob(self.root_dir+"datasets/final_soop_dataset_small/adc_registered/*.nii.gz"))

        tests_anomaly_masks = glob.glob(self.root_dir+"datasets/final_soop_dataset_small/masks_combined_registered/*.nii.gz")


        images_to_exclude = []
        with open(self.root_dir+"AnoDiffExperiments/data_splits_lists/final_soop_dataset_small/exclude.csv", 'r') as f:
            for line in f:
                images_to_exclude.append(line.strip())

        with open(self.root_dir+"AnoDiffExperiments/data_splits_lists/final_soop_dataset_small/exclude_non_axial_thick_slices.csv", 'r') as f:
            for line in f:
                images_to_exclude.append(line.strip())
        
        test_anomaly_transforms = define_instance(self.args, "val_transforms")

        # final lists of images and masks path after excluding the bad images
        self.test_anomaly_large_images = [path for path in test_anomaly_images if os.path.basename(path).split('.')[0] not in images_to_exclude and os.path.basename(path).split('.')[0] in large_group]        
        self.large_group_masks = [path for path in tests_anomaly_masks if os.path.basename(path).split('.')[0] not in images_to_exclude and os.path.basename(path).split('.')[0] in large_group]

        self.test_anomaly_large_images = sorted(self.test_anomaly_large_images, key=lambda x: os.path.basename(x).split('.')[0])
        self.large_group_masks = sorted(self.large_group_masks, key=lambda x: os.path.basename(x).split('.')[0])

        if num_images_to_load > 0:
            self.test_anomaly_large_images = self.test_anomaly_large_images[:num_images_to_load]
            self.large_group_masks = self.large_group_masks[:num_images_to_load]

        # dataloaders
        # each group of images is split into two halves:
        # select params half: used to select the best noise timestep value, best threshold etc
        # metrics half: used to compute the final scores (e.g DICE) with these best values.

        # images
        test_anomaly_large_ds = CacheDataset(data=self.test_anomaly_large_images, transform=test_anomaly_transforms)
        
        # large group
        self.test_anomaly_large_loader_select_params = DataLoader( 
            _subset_first_half(test_anomaly_large_ds), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        self.test_anomaly_large_images_select_params = self.test_anomaly_large_images[:len(self.test_anomaly_large_images)//2]

        self.test_anomaly_large_loader_metrics = DataLoader(       
            _subset_second_half(test_anomaly_large_ds), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        self.test_anomaly_large_images_metrics = self.test_anomaly_large_images[len(self.test_anomaly_large_images)//2:]

        # masks
        self.test_masks_large_ds = CacheDataset(data=self.large_group_masks, transform=masks_transforms)
       
        # large group
        self.test_masks_large_loader_select_params = DataLoader(
            _subset_first_half(self.test_masks_large_ds), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        self.test_masks_large_loader_metrics = DataLoader(
            _subset_second_half(self.test_masks_large_ds), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )


    def len_large_group(self):
        return len(self.test_anomaly_large_images)
    
    def first(self):
        return first(self.test_anomaly_large_loader_select_params)
    
    def get_anomaly_loader_select_params(self, group):
        if group == "large":
            return self.test_anomaly_large_loader_select_params
        else:
            raise ValueError("Invalid group name. Must be 'large', 'medium', or 'small'.")
    
    def get_anomaly_loader_metrics(self, group):
        if group == "large":
            return self.test_anomaly_large_loader_metrics
        else:
            raise ValueError("Invalid group name. Must be 'large', 'medium', or 'small'.")
    
    def get_masks_loader_select_params(self, group):
        if group == "large":
            return self.test_masks_large_loader_select_params
        else:
            raise ValueError("Invalid group name. Must be 'large', 'medium', or 'small'.")
    
    def get_masks_loader_metrics(self, group):
        if group == "large":
            return self.test_masks_large_loader_metrics
        else:
            raise ValueError("Invalid group name. Must be 'large', 'medium', or 'small'.")

    def get_anomaly_images_select_params(self, group):
        if group == "large":
            return self.test_anomaly_large_images_select_params
        else:
            raise ValueError("Invalid group name. Must be 'large', 'medium', or 'small'.")
    
    def get_anomaly_images_metrics(self, group):
        if group == "large":
            return self.test_anomaly_large_images_metrics
        else:
            raise ValueError("Invalid group name. Must be 'large', 'medium', or 'small'.")

class SOOP_Fast():
    def __init__(self, 
                 args,
                 batch_size=64,
                 num_workers=4,
                ):
        
        self.args = args
        self.root_dir = args.root_dir
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
    

        masks_transforms = transforms.Compose(
            [
                transforms.LoadImage(),
                transforms.EnsureChannelFirst(),
                transforms.ResizeWithPadOrCrop(spatial_size=(args.image_size, args.image_size, args.image_size)),
                custom_transforms.SetBackgroundToZero()
            ]
        )

        # Only keep first 8 subjects from large group
        large_group = ['sub-1010', 'sub-1013', 'sub-1015', 'sub-1032', 'sub-1035', 'sub-1039', 'sub-1041', 'sub-1045']

        if "flair" in self.args.dataset["name"].lower():
            test_anomaly_images = sorted(glob.glob(self.root_dir+"datasets/final_soop_dataset_small/flair_registered/*.nii.gz"))
        elif "adc" in self.args.dataset["name"].lower():
            test_anomaly_images = sorted(glob.glob(self.root_dir+"datasets/final_soop_dataset_small/adc_registered/*.nii.gz"))

        tests_anomaly_masks = glob.glob(self.root_dir+"datasets/final_soop_dataset_small/masks_combined_registered/*.nii.gz")

        images_to_exclude = []
        with open(self.root_dir+"AnoDiffExperiments/data_splits_lists/final_soop_dataset_small/exclude.csv", 'r') as f:
            for line in f:
                images_to_exclude.append(line.strip())

        with open(self.root_dir+"AnoDiffExperiments/data_splits_lists/final_soop_dataset_small/exclude_non_axial_thick_slices.csv", 'r') as f:
            for line in f:
                images_to_exclude.append(line.strip())
        
        test_anomaly_transforms = define_instance(self.args, "val_transforms")

        # Filter for large group only
        self.test_anomaly_large_images = [path for path in test_anomaly_images if os.path.basename(path).split('.')[0] not in images_to_exclude and os.path.basename(path).split('.')[0] in large_group]        
        self.large_group_masks = [path for path in tests_anomaly_masks if os.path.basename(path).split('.')[0] not in images_to_exclude and os.path.basename(path).split('.')[0] in large_group]

        self.test_anomaly_large_images = sorted(self.test_anomaly_large_images, key=lambda x: os.path.basename(x).split('.')[0])
        self.large_group_masks = sorted(self.large_group_masks, key=lambda x: os.path.basename(x).split('.')[0])

        # dataloaders - split 8 images into 4 and 4
        test_anomaly_large_ds = CacheDataset(data=self.test_anomaly_large_images, transform=test_anomaly_transforms)
        
        # large group - select params (first 4)
        self.test_anomaly_large_loader_select_params = DataLoader( 
            _subset_range(test_anomaly_large_ds, 0, 4), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        self.test_anomaly_large_images_select_params = self.test_anomaly_large_images[:4]

        # large group - metrics (last 4)
        self.test_anomaly_large_loader_metrics = DataLoader(       
            _subset_range(test_anomaly_large_ds, 4, len(test_anomaly_large_ds)), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        self.test_anomaly_large_images_metrics = self.test_anomaly_large_images[4:]
        
        # masks
        self.test_masks_large_ds = CacheDataset(data=self.large_group_masks, transform=masks_transforms)

        self.test_masks_large_loader_select_params = DataLoader(
            _subset_range(self.test_masks_large_ds, 0, 4), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        self.test_masks_large_loader_metrics = DataLoader(
            _subset_range(self.test_masks_large_ds, 4, len(self.test_masks_large_ds)), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )

    def len_large_group(self):
        return len(self.test_anomaly_large_images)

    def first(self):
        return first(self.test_anomaly_large_loader_select_params)


class SOOP_Fast_adc_flair():
    def __init__(self, 
                 args,
                 batch_size=64,
                 num_workers=4,
                ):
        
        self.args = args
        self.root_dir = args.root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
    

        masks_transforms = monai.transforms.Compose(
            [
                monai.transforms.LoadImage(),
                monai.transforms.EnsureChannelFirst(),
                monai.transforms.ResizeWithPadOrCrop(spatial_size=(args.image_size, args.image_size, args.image_size)),
                custom_transforms.SetBackgroundToZero()
            ]
        )

        # Only keep first 8 subjects from large group
        large_group = ['sub-1010', 'sub-1013', 'sub-1015', 'sub-1032', 'sub-1035', 'sub-1039', 'sub-1041', 'sub-1045']

        
        test_anomaly_images_flair = sorted(glob.glob(self.root_dir+"datasets/final_soop_dataset_small/flair_registered/*.nii.gz"))
        
        test_anomaly_images_adc = sorted(glob.glob(self.root_dir+"datasets/final_soop_dataset_small/adc_registered/*.nii.gz"))

        tests_anomaly_masks = glob.glob(self.root_dir+"datasets/final_soop_dataset_small/masks_combined_registered/*.nii.gz")

        images_to_exclude = []
        with open(self.root_dir+"AnoDiffExperiments/data_splits_lists/final_soop_dataset_small/exclude.csv", 'r') as f:
            for line in f:
                images_to_exclude.append(line.strip())

        with open(self.root_dir+"AnoDiffExperiments/data_splits_lists/final_soop_dataset_small/exclude_non_axial_thick_slices.csv", 'r') as f:
            for line in f:
                images_to_exclude.append(line.strip())
        
        test_anomaly_transforms = define_instance(self.args, "val_transforms")

        # ------------ Masks ------------

        self.large_group_masks = [path for path in tests_anomaly_masks if os.path.basename(path).split('.')[0] not in images_to_exclude and os.path.basename(path).split('.')[0] in large_group]
        self.large_group_masks = sorted(self.large_group_masks, key=lambda x: os.path.basename(x).split('.')[0])

        # masks
        self.test_masks_large_ds = CacheDataset(data=self.large_group_masks, transform=masks_transforms)

        self.test_masks_large_ds = CacheDataset(data=self.large_group_masks, transform=masks_transforms)

        self.test_masks_large_loader_select_params = DataLoader(
            _subset_range(self.test_masks_large_ds, 0, 4), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        self.test_masks_large_loader_metrics = DataLoader(
            _subset_range(self.test_masks_large_ds, 4, len(self.test_masks_large_ds)), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )

        # ------------ ADC + FLAIR ------------

        # Filter for large group only
        self.test_anomaly_large_images_adc = [path for path in test_anomaly_images_adc if os.path.basename(path).split('.')[0] not in images_to_exclude and os.path.basename(path).split('.')[0] in large_group]        
        self.test_anomaly_large_images_adc = sorted(self.test_anomaly_large_images_adc, key=lambda x: os.path.basename(x).split('.')[0])

        self.test_anomaly_large_images_flair = [path for path in test_anomaly_images_flair if os.path.basename(path).split('.')[0] not in images_to_exclude and os.path.basename(path).split('.')[0] in large_group]        
        self.test_anomaly_large_images_flair = sorted(self.test_anomaly_large_images_flair, key=lambda x: os.path.basename(x).split('.')[0])

        self.datalist = [
            {"adc": adc_path, "flair": flair_path}
            for adc_path, flair_path in zip(self.test_anomaly_large_images_adc, self.test_anomaly_large_images_flair)
        ]

        transforms_def = define_instance(args, "val_transforms")

        ano_transforms = monai.transforms.Compose([
            *transforms_def.transforms,
            monai.transforms.Lambda(func=lambda x: x["image"]),
        ])


        # dataloaders - split 8 images into 4 and 4
        test_anomaly_large_ds = CacheDataset(data=self.datalist, transform=ano_transforms)
        
        # large group - select params (first 4)
        self.test_anomaly_large_loader_select_params = DataLoader( 
            _subset_range(test_anomaly_large_ds, 0, 4), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        self.test_anomaly_large_images_select_params_adc = self.test_anomaly_large_images_adc[:4]
        self.test_anomaly_large_images_select_params_flair = self.test_anomaly_large_images_flair[:4]

        # large group - metrics (last 4)
        self.test_anomaly_large_loader_metrics = DataLoader(       
            _subset_range(test_anomaly_large_ds, 4, len(test_anomaly_large_ds)), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )

        self.test_anomaly_large_images_metrics_adc = self.test_anomaly_large_images_adc[:4]
        self.test_anomaly_large_images_metrics_flair = self.test_anomaly_large_images_flair[:4]



    def len_large_group(self):
        return len(self.test_anomaly_large_images)

    def first(self):
        return first(self.test_anomaly_large_loader_select_params)


class SOOP_Fast_adc_flair_t1w():
    def __init__(self, 
                 args,
                 batch_size=64,
                 num_workers=4,
                ):
        
        self.args = args
        self.root_dir = args.root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
    

        masks_transforms = monai.transforms.Compose(
            [
                monai.transforms.LoadImage(),
                monai.transforms.EnsureChannelFirst(),
                monai.transforms.ResizeWithPadOrCrop(spatial_size=(args.image_size, args.image_size, args.image_size)),
                custom_transforms.SetBackgroundToZero()
            ]
        )

        # Only keep first 8 subjects from large group
        large_group = ['sub-1010', 'sub-1013', 'sub-1015', 'sub-1032', 'sub-1035', 'sub-1039', 'sub-1041', 'sub-1045']

        
        test_anomaly_images_flair = sorted(glob.glob(self.root_dir+"datasets/final_soop_dataset_small/flair_registered/*.nii.gz"))
        
        test_anomaly_images_adc = sorted(glob.glob(self.root_dir+"datasets/final_soop_dataset_small/adc_registered/*.nii.gz"))

        test_anomaly_images_t1w = sorted(glob.glob(self.root_dir+"datasets/final_soop_dataset_small/t1w_registered/*.nii.gz"))

        tests_anomaly_masks = glob.glob(self.root_dir+"datasets/final_soop_dataset_small/masks_combined_registered/*.nii.gz")

        images_to_exclude = []
        with open(self.root_dir+"AnoDiffExperiments/data_splits_lists/final_soop_dataset_small/exclude.csv", 'r') as f:
            for line in f:
                images_to_exclude.append(line.strip())

        with open(self.root_dir+"AnoDiffExperiments/data_splits_lists/final_soop_dataset_small/exclude_non_axial_thick_slices.csv", 'r') as f:
            for line in f:
                images_to_exclude.append(line.strip())
        
        test_anomaly_transforms = define_instance(self.args, "val_transforms")

        # ------------ Masks ------------

        self.large_group_masks = [path for path in tests_anomaly_masks if os.path.basename(path).split('.')[0] not in images_to_exclude and os.path.basename(path).split('.')[0] in large_group]
        self.large_group_masks = sorted(self.large_group_masks, key=lambda x: os.path.basename(x).split('.')[0])

        # masks
        self.test_masks_large_ds = CacheDataset(data=self.large_group_masks, transform=masks_transforms)

        self.test_masks_large_ds = CacheDataset(data=self.large_group_masks, transform=masks_transforms)

        self.test_masks_large_loader_select_params = DataLoader(
            _subset_range(self.test_masks_large_ds, 0, 4), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        self.test_masks_large_loader_metrics = DataLoader(
            _subset_range(self.test_masks_large_ds, 4, len(self.test_masks_large_ds)), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )

        # ------------ ADC + FLAIR + T1w ------------

        # Filter for large group only
        self.test_anomaly_large_images_adc = [path for path in test_anomaly_images_adc if os.path.basename(path).split('.')[0] not in images_to_exclude and os.path.basename(path).split('.')[0] in large_group]        
        self.test_anomaly_large_images_adc = sorted(self.test_anomaly_large_images_adc, key=lambda x: os.path.basename(x).split('.')[0])

        self.test_anomaly_large_images_flair = [path for path in test_anomaly_images_flair if os.path.basename(path).split('.')[0] not in images_to_exclude and os.path.basename(path).split('.')[0] in large_group]        
        self.test_anomaly_large_images_flair = sorted(self.test_anomaly_large_images_flair, key=lambda x: os.path.basename(x).split('.')[0])

        self.test_anomaly_large_images_t1w = [path for path in test_anomaly_images_t1w if os.path.basename(path).split('.')[0] not in images_to_exclude and os.path.basename(path).split('.')[0] in large_group]
        self.test_anomaly_large_images_t1w = sorted(self.test_anomaly_large_images_t1w, key=lambda x: os.path.basename(x).split('.')[0])

        self.datalist = [
            {"adc": adc_path, "flair": flair_path, "t1w": t1w_path}
            for adc_path, flair_path, t1w_path in zip(self.test_anomaly_large_images_adc, self.test_anomaly_large_images_flair, self.test_anomaly_large_images_t1w)
        ]

        transforms_def = define_instance(args, "val_transforms")

        ano_transforms = monai.transforms.Compose([
            *transforms_def.transforms,
            monai.transforms.Lambda(func=lambda x: x["image"]),
        ])


        # dataloaders - split 8 images into 4 and 4
        test_anomaly_large_ds = CacheDataset(data=self.datalist, transform=ano_transforms)
        
        # large group - select params (first 4)
        self.test_anomaly_large_loader_select_params = DataLoader( 
            _subset_range(test_anomaly_large_ds, 0, 4), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        self.test_anomaly_large_images_select_params_adc = self.test_anomaly_large_images_adc[:4]
        self.test_anomaly_large_images_select_params_flair = self.test_anomaly_large_images_flair[:4]
        self.test_anomaly_large_images_select_params_t1w = self.test_anomaly_large_images_t1w[:4]
        # large group - metrics (last 4)
        self.test_anomaly_large_loader_metrics = DataLoader(       
            _subset_range(test_anomaly_large_ds, 4, len(test_anomaly_large_ds)), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )

        self.test_anomaly_large_images_metrics_adc = self.test_anomaly_large_images_adc[:4]
        self.test_anomaly_large_images_metrics_flair = self.test_anomaly_large_images_flair[:4]
        self.test_anomaly_large_images_metrics_t1w = self.test_anomaly_large_images_t1w[:4]



    def len_large_group(self):
        return len(self.test_anomaly_large_images)

    def first(self):
        return first(self.test_anomaly_large_loader_select_params)


class SOOP_adc_flair_t1w():
    def __init__(self,
                 args,
                 batch_size=64,
                 num_workers=4,
                ):

        self.args = args
        self.root_dir = args.root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        masks_transforms = monai.transforms.Compose(
            [
                monai.transforms.LoadImage(),
                monai.transforms.EnsureChannelFirst(),
                monai.transforms.ResizeWithPadOrCrop(spatial_size=(args.image_size, args.image_size, args.image_size)),
                custom_transforms.SetBackgroundToZero()
            ]
        )

        # Reuse the existing SOOP split logic (large / medium / small), then build multimodal triplets per group.
        base_soop = SOOP(args, batch_size=batch_size, num_workers=num_workers)

        adc_images = sorted(glob.glob(self.root_dir + "datasets/final_soop_dataset_small/adc_registered/*.nii.gz"))
        flair_images = sorted(glob.glob(self.root_dir + "datasets/final_soop_dataset_small/flair_registered/*.nii.gz"))
        t1w_images = sorted(glob.glob(self.root_dir + "datasets/final_soop_dataset_small/t1w_registered/*.nii.gz"))
        masks_images = sorted(glob.glob(self.root_dir + "datasets/final_soop_dataset_small/masks_combined_registered/*.nii.gz"))

        adc_by_id = {os.path.basename(path).split('.')[0]: path for path in adc_images}
        flair_by_id = {os.path.basename(path).split('.')[0]: path for path in flair_images}
        t1w_by_id = {os.path.basename(path).split('.')[0]: path for path in t1w_images}
        masks_by_id = {os.path.basename(path).split('.')[0]: path for path in masks_images}

        def build_group(group_paths):
            group_ids = [os.path.basename(path).split('.')[0] for path in group_paths]
            valid_ids = [
                subject_id for subject_id in group_ids
                if subject_id in adc_by_id and subject_id in flair_by_id and subject_id in t1w_by_id and subject_id in masks_by_id
            ]

            adc_group = [adc_by_id[subject_id] for subject_id in valid_ids]
            flair_group = [flair_by_id[subject_id] for subject_id in valid_ids]
            t1w_group = [t1w_by_id[subject_id] for subject_id in valid_ids]
            masks_group = [masks_by_id[subject_id] for subject_id in valid_ids]

            datalist_group = [
                {"adc": adc_path, "flair": flair_path, "t1w": t1w_path}
                for adc_path, flair_path, t1w_path in zip(adc_group, flair_group, t1w_group)
            ]
            return datalist_group, adc_group, flair_group, t1w_group, masks_group

        (self.test_anomaly_large_images,
         self.test_anomaly_large_images_adc,
         self.test_anomaly_large_images_flair,
         self.test_anomaly_large_images_t1w,
         self.large_group_masks) = build_group(base_soop.test_anomaly_large_images)

        (self.test_anomaly_medium_images,
         self.test_anomaly_medium_images_adc,
         self.test_anomaly_medium_images_flair,
         self.test_anomaly_medium_images_t1w,
         self.medium_group_masks) = build_group(base_soop.test_anomaly_medium_images)

        (self.test_anomaly_small_images,
         self.test_anomaly_small_images_adc,
         self.test_anomaly_small_images_flair,
         self.test_anomaly_small_images_t1w,
         self.small_group_masks) = build_group(base_soop.test_anomaly_small_images)

        transforms_def = define_instance(args, "val_transforms")
        ano_transforms = monai.transforms.Compose([
            *transforms_def.transforms,
            monai.transforms.Lambda(func=lambda x: x["image"]),
        ])

        test_anomaly_large_ds = CacheDataset(data=self.test_anomaly_large_images, transform=ano_transforms)
        test_anomaly_medium_ds = CacheDataset(data=self.test_anomaly_medium_images, transform=ano_transforms)
        test_anomaly_small_ds = CacheDataset(data=self.test_anomaly_small_images, transform=ano_transforms)

        self.test_masks_large_ds = CacheDataset(data=self.large_group_masks, transform=masks_transforms)
        self.test_masks_medium_ds = CacheDataset(data=self.medium_group_masks, transform=masks_transforms)
        self.test_masks_small_ds = CacheDataset(data=self.small_group_masks, transform=masks_transforms)

        large_split = len(test_anomaly_large_ds) // 2
        medium_split = len(test_anomaly_medium_ds) // 2
        small_split = len(test_anomaly_small_ds) // 2

        self.test_anomaly_large_loader_select_params = DataLoader(
            _subset_range(test_anomaly_large_ds, 0, large_split), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        self.test_anomaly_large_loader_metrics = DataLoader(
            _subset_range(test_anomaly_large_ds, large_split, len(test_anomaly_large_ds)), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )

        self.test_anomaly_medium_loader_select_params = DataLoader(
            _subset_range(test_anomaly_medium_ds, 0, medium_split), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        self.test_anomaly_medium_loader_metrics = DataLoader(
            _subset_range(test_anomaly_medium_ds, medium_split, len(test_anomaly_medium_ds)), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )

        self.test_anomaly_small_loader_select_params = DataLoader(
            _subset_range(test_anomaly_small_ds, 0, small_split), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        self.test_anomaly_small_loader_metrics = DataLoader(
            _subset_range(test_anomaly_small_ds, small_split, len(test_anomaly_small_ds)), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )

        self.test_masks_large_loader_select_params = DataLoader(
            _subset_range(self.test_masks_large_ds, 0, large_split), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        self.test_masks_large_loader_metrics = DataLoader(
            _subset_range(self.test_masks_large_ds, large_split, len(self.test_masks_large_ds)), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )

        self.test_masks_medium_loader_select_params = DataLoader(
            _subset_range(self.test_masks_medium_ds, 0, medium_split), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        self.test_masks_medium_loader_metrics = DataLoader(
            _subset_range(self.test_masks_medium_ds, medium_split, len(self.test_masks_medium_ds)), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )

        self.test_masks_small_loader_select_params = DataLoader(
            _subset_range(self.test_masks_small_ds, 0, small_split), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        self.test_masks_small_loader_metrics = DataLoader(
            _subset_range(self.test_masks_small_ds, small_split, len(self.test_masks_small_ds)), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )

        self.test_anomaly_large_images_select_params = self.test_anomaly_large_images[:large_split]
        self.test_anomaly_large_images_metrics = self.test_anomaly_large_images[large_split:]
        self.test_anomaly_medium_images_select_params = self.test_anomaly_medium_images[:medium_split]
        self.test_anomaly_medium_images_metrics = self.test_anomaly_medium_images[medium_split:]
        self.test_anomaly_small_images_select_params = self.test_anomaly_small_images[:small_split]
        self.test_anomaly_small_images_metrics = self.test_anomaly_small_images[small_split:]

    def len_large_group(self):
        return len(self.test_anomaly_large_images)

    def len_medium_group(self):
        return len(self.test_anomaly_medium_images)

    def len_small_group(self):
        return len(self.test_anomaly_small_images)

    def first(self):
        return first(self.test_anomaly_large_loader_select_params)

    def get_anomaly_loader_select_params(self, group):
        if group == "large":
            return self.test_anomaly_large_loader_select_params
        elif group == "medium":
            return self.test_anomaly_medium_loader_select_params
        elif group == "small":
            return self.test_anomaly_small_loader_select_params
        else:
            raise ValueError("Invalid group name. Must be 'large', 'medium', or 'small'.")

    def get_anomaly_loader_metrics(self, group):
        if group == "large":
            return self.test_anomaly_large_loader_metrics
        elif group == "medium":
            return self.test_anomaly_medium_loader_metrics
        elif group == "small":
            return self.test_anomaly_small_loader_metrics
        else:
            raise ValueError("Invalid group name. Must be 'large', 'medium', or 'small'.")

    def get_masks_loader_select_params(self, group):
        if group == "large":
            return self.test_masks_large_loader_select_params
        elif group == "medium":
            return self.test_masks_medium_loader_select_params
        elif group == "small":
            return self.test_masks_small_loader_select_params
        else:
            raise ValueError("Invalid group name. Must be 'large', 'medium', or 'small'.")

    def get_masks_loader_metrics(self, group):
        if group == "large":
            return self.test_masks_large_loader_metrics
        elif group == "medium":
            return self.test_masks_medium_loader_metrics
        elif group == "small":
            return self.test_masks_small_loader_metrics
        else:
            raise ValueError("Invalid group name. Must be 'large', 'medium', or 'small'.")

    def get_anomaly_images_select_params(self, group):
        if group == "large":
            return self.test_anomaly_large_images_select_params
        elif group == "medium":
            return self.test_anomaly_medium_images_select_params
        elif group == "small":
            return self.test_anomaly_small_images_select_params
        else:
            raise ValueError("Invalid group name. Must be 'large', 'medium', or 'small'.")

    def get_anomaly_images_metrics(self, group):
        if group == "large":
            return self.test_anomaly_large_images_metrics
        elif group == "medium":
            return self.test_anomaly_medium_images_metrics
        elif group == "small":
            return self.test_anomaly_small_images_metrics
        else:
            raise ValueError("Invalid group name. Must be 'large', 'medium', or 'small'.")

class ISLES():

    def __init__(self, 
                 args,
                 batch_size=64,
                 num_workers=4,
                ):
        
        self.args = args
        self.root_dir = args.root_dir
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers


        masks_transforms = transforms.Compose(
            [
                transforms.LoadImage(),
                transforms.EnsureChannelFirst(),
                transforms.ResizeWithPadOrCrop(spatial_size=(args.image_size, args.image_size, args.image_size)),
                custom_transforms.SetBackgroundToZero()
            ]
        )

        #TODO renommer les masks et les images pour qu'ils aient exactement le meme nom
        large_group = ['sub-strokecase0023_ses-0001_msk.nii.gz', 'sub-strokecase0031_ses-0001_msk.nii.gz', 'sub-strokecase0047_ses-0001_msk.nii.gz', 'sub-strokecase0048_ses-0001_msk.nii.gz', 'sub-strokecase0062_ses-0001_msk.nii.gz', 'sub-strokecase0066_ses-0001_msk.nii.gz', 'sub-strokecase0081_ses-0001_msk.nii.gz', 'sub-strokecase0083_ses-0001_msk.nii.gz', 'sub-strokecase0087_ses-0001_msk.nii.gz', 'sub-strokecase0091_ses-0001_msk.nii.gz', 'sub-strokecase0123_ses-0001_msk.nii.gz', 'sub-strokecase0161_ses-0001_msk.nii.gz', 'sub-strokecase0162_ses-0001_msk.nii.gz', 'sub-strokecase0171_ses-0001_msk.nii.gz', 'sub-strokecase0176_ses-0001_msk.nii.gz', 'sub-strokecase0201_ses-0001_msk.nii.gz', 'sub-strokecase0211_ses-0001_msk.nii.gz', 'sub-strokecase0222_ses-0001_msk.nii.gz', 'sub-strokecase0223_ses-0001_msk.nii.gz', 'sub-strokecase0023_ses-0001_msk.nii.gz', 'sub-strokecase0031_ses-0001_msk.nii.gz', 'sub-strokecase0047_ses-0001_msk.nii.gz', 'sub-strokecase0048_ses-0001_msk.nii.gz', 'sub-strokecase0062_ses-0001_msk.nii.gz', 'sub-strokecase0066_ses-0001_msk.nii.gz', 'sub-strokecase0081_ses-0001_msk.nii.gz', 'sub-strokecase0083_ses-0001_msk.nii.gz', 'sub-strokecase0087_ses-0001_msk.nii.gz', 'sub-strokecase0091_ses-0001_msk.nii.gz', 'sub-strokecase0123_ses-0001_msk.nii.gz', 'sub-strokecase0161_ses-0001_msk.nii.gz', 'sub-strokecase0162_ses-0001_msk.nii.gz', 'sub-strokecase0171_ses-0001_msk.nii.gz', 'sub-strokecase0176_ses-0001_msk.nii.gz', 'sub-strokecase0201_ses-0001_msk.nii.gz', 'sub-strokecase0211_ses-0001_msk.nii.gz', 'sub-strokecase0222_ses-0001_msk.nii.gz', 'sub-strokecase0223_ses-0001_msk.nii.gz', 'sub-strokecase0230_ses-0001_msk.nii.gz', 'sub-strokecase0237_ses-0001_msk.nii.gz', 'sub-strokecase0240_ses-0001_msk.nii.gz', 'sub-strokecase0246_ses-0001_msk.nii.gz']
        self.large_group_adc_images = [self.root_dir+"datasets/final_adc_dataset_small/ISLES_registered/"+filename.replace("msk", "adc") for filename in large_group]
        self.large_group_flair_images = [self.root_dir+"datasets/final_flair_dataset_small/isles_registered/"+filename.replace("msk", "FLAIR") for filename in large_group]
        self.large_group_flair_images = [path for path in self.large_group_flair_images if "0222_ses-0001" not in path]
        self.large_group_masks = [self.root_dir+"datasets/final_adc_dataset_small/ISLES_masks_registered/"+filename for filename in large_group]

        medium_group = ['sub-strokecase0001_ses-0001_msk.nii.gz', 'sub-strokecase0003_ses-0001_msk.nii.gz', 'sub-strokecase0011_ses-0001_msk.nii.gz', 'sub-strokecase0013_ses-0001_msk.nii.gz', 'sub-strokecase0015_ses-0001_msk.nii.gz', 'sub-strokecase0021_ses-0001_msk.nii.gz', 'sub-strokecase0027_ses-0001_msk.nii.gz', 'sub-strokecase0033_ses-0001_msk.nii.gz', 'sub-strokecase0039_ses-0001_msk.nii.gz', 'sub-strokecase0043_ses-0001_msk.nii.gz', 'sub-strokecase0052_ses-0001_msk.nii.gz', 'sub-strokecase0057_ses-0001_msk.nii.gz', 'sub-strokecase0065_ses-0001_msk.nii.gz', 'sub-strokecase0085_ses-0001_msk.nii.gz', 'sub-strokecase0092_ses-0001_msk.nii.gz', 'sub-strokecase0101_ses-0001_msk.nii.gz', 'sub-strokecase0102_ses-0001_msk.nii.gz', 'sub-strokecase0114_ses-0001_msk.nii.gz', 'sub-strokecase0116_ses-0001_msk.nii.gz', 'sub-strokecase0120_ses-0001_msk.nii.gz', 'sub-strokecase0122_ses-0001_msk.nii.gz', 'sub-strokecase0124_ses-0001_msk.nii.gz', 'sub-strokecase0127_ses-0001_msk.nii.gz', 'sub-strokecase0140_ses-0001_msk.nii.gz', 'sub-strokecase0146_ses-0001_msk.nii.gz', 'sub-strokecase0153_ses-0001_msk.nii.gz', 'sub-strokecase0154_ses-0001_msk.nii.gz', 'sub-strokecase0155_ses-0001_msk.nii.gz', 'sub-strokecase0164_ses-0001_msk.nii.gz', 'sub-strokecase0165_ses-0001_msk.nii.gz', 'sub-strokecase0166_ses-0001_msk.nii.gz', 'sub-strokecase0168_ses-0001_msk.nii.gz', 'sub-strokecase0178_ses-0001_msk.nii.gz', 'sub-strokecase0179_ses-0001_msk.nii.gz', 'sub-strokecase0180_ses-0001_msk.nii.gz', 'sub-strokecase0186_ses-0001_msk.nii.gz', 'sub-strokecase0188_ses-0001_msk.nii.gz', 'sub-strokecase0189_ses-0001_msk.nii.gz', 'sub-strokecase0190_ses-0001_msk.nii.gz', 'sub-strokecase0191_ses-0001_msk.nii.gz', 'sub-strokecase0192_ses-0001_msk.nii.gz', 'sub-strokecase0194_ses-0001_msk.nii.gz', 'sub-strokecase0195_ses-0001_msk.nii.gz', 'sub-strokecase0199_ses-0001_msk.nii.gz', 'sub-strokecase0204_ses-0001_msk.nii.gz', 'sub-strokecase0206_ses-0001_msk.nii.gz', 'sub-strokecase0207_ses-0001_msk.nii.gz', 'sub-strokecase0208_ses-0001_msk.nii.gz', 'sub-strokecase0209_ses-0001_msk.nii.gz', 'sub-strokecase0215_ses-0001_msk.nii.gz', 'sub-strokecase0219_ses-0001_msk.nii.gz', 'sub-strokecase0220_ses-0001_msk.nii.gz', 'sub-strokecase0001_ses-0001_msk.nii.gz', 'sub-strokecase0003_ses-0001_msk.nii.gz', 'sub-strokecase0011_ses-0001_msk.nii.gz', 'sub-strokecase0013_ses-0001_msk.nii.gz', 'sub-strokecase0015_ses-0001_msk.nii.gz', 'sub-strokecase0021_ses-0001_msk.nii.gz', 'sub-strokecase0027_ses-0001_msk.nii.gz', 'sub-strokecase0033_ses-0001_msk.nii.gz', 'sub-strokecase0039_ses-0001_msk.nii.gz', 'sub-strokecase0043_ses-0001_msk.nii.gz', 'sub-strokecase0052_ses-0001_msk.nii.gz', 'sub-strokecase0057_ses-0001_msk.nii.gz', 'sub-strokecase0065_ses-0001_msk.nii.gz', 'sub-strokecase0085_ses-0001_msk.nii.gz', 'sub-strokecase0092_ses-0001_msk.nii.gz', 'sub-strokecase0101_ses-0001_msk.nii.gz', 'sub-strokecase0102_ses-0001_msk.nii.gz', 'sub-strokecase0114_ses-0001_msk.nii.gz', 'sub-strokecase0116_ses-0001_msk.nii.gz', 'sub-strokecase0120_ses-0001_msk.nii.gz', 'sub-strokecase0122_ses-0001_msk.nii.gz', 'sub-strokecase0124_ses-0001_msk.nii.gz', 'sub-strokecase0127_ses-0001_msk.nii.gz', 'sub-strokecase0140_ses-0001_msk.nii.gz', 'sub-strokecase0146_ses-0001_msk.nii.gz', 'sub-strokecase0153_ses-0001_msk.nii.gz', 'sub-strokecase0154_ses-0001_msk.nii.gz', 'sub-strokecase0155_ses-0001_msk.nii.gz', 'sub-strokecase0164_ses-0001_msk.nii.gz', 'sub-strokecase0165_ses-0001_msk.nii.gz', 'sub-strokecase0166_ses-0001_msk.nii.gz', 'sub-strokecase0168_ses-0001_msk.nii.gz', 'sub-strokecase0178_ses-0001_msk.nii.gz', 'sub-strokecase0179_ses-0001_msk.nii.gz', 'sub-strokecase0180_ses-0001_msk.nii.gz', 'sub-strokecase0186_ses-0001_msk.nii.gz', 'sub-strokecase0188_ses-0001_msk.nii.gz', 'sub-strokecase0189_ses-0001_msk.nii.gz', 'sub-strokecase0190_ses-0001_msk.nii.gz', 'sub-strokecase0191_ses-0001_msk.nii.gz', 'sub-strokecase0192_ses-0001_msk.nii.gz', 'sub-strokecase0194_ses-0001_msk.nii.gz', 'sub-strokecase0195_ses-0001_msk.nii.gz', 'sub-strokecase0199_ses-0001_msk.nii.gz', 'sub-strokecase0204_ses-0001_msk.nii.gz', 'sub-strokecase0206_ses-0001_msk.nii.gz', 'sub-strokecase0207_ses-0001_msk.nii.gz', 'sub-strokecase0208_ses-0001_msk.nii.gz', 'sub-strokecase0209_ses-0001_msk.nii.gz', 'sub-strokecase0215_ses-0001_msk.nii.gz', 'sub-strokecase0219_ses-0001_msk.nii.gz', 'sub-strokecase0220_ses-0001_msk.nii.gz', 'sub-strokecase0226_ses-0001_msk.nii.gz', 'sub-strokecase0227_ses-0001_msk.nii.gz', 'sub-strokecase0236_ses-0001_msk.nii.gz', 'sub-strokecase0238_ses-0001_msk.nii.gz', 'sub-strokecase0243_ses-0001_msk.nii.gz', 'sub-strokecase0245_ses-0001_msk.nii.gz', 'sub-strokecase0248_ses-0001_msk.nii.gz']
        self.medium_group_adc_images = [self.root_dir+"datasets/final_adc_dataset_small/ISLES_registered/"+filename.replace("msk", "adc") for filename in medium_group]
        self.medium_group_flair_images = [self.root_dir+"datasets/final_flair_dataset_small/isles_registered/"+filename.replace("msk", "FLAIR") for filename in medium_group]
        self.medium_group_masks = [self.root_dir+"datasets/final_adc_dataset_small/ISLES_masks_registered/"+filename for filename in medium_group]

        small_group = ['sub-strokecase0004_ses-0001_msk.nii.gz', 'sub-strokecase0009_ses-0001_msk.nii.gz', 'sub-strokecase0010_ses-0001_msk.nii.gz', 'sub-strokecase0017_ses-0001_msk.nii.gz', 'sub-strokecase0022_ses-0001_msk.nii.gz', 'sub-strokecase0024_ses-0001_msk.nii.gz', 'sub-strokecase0026_ses-0001_msk.nii.gz', 'sub-strokecase0036_ses-0001_msk.nii.gz', 'sub-strokecase0038_ses-0001_msk.nii.gz', 'sub-strokecase0040_ses-0001_msk.nii.gz', 'sub-strokecase0041_ses-0001_msk.nii.gz', 'sub-strokecase0049_ses-0001_msk.nii.gz', 'sub-strokecase0053_ses-0001_msk.nii.gz', 'sub-strokecase0054_ses-0001_msk.nii.gz', 'sub-strokecase0056_ses-0001_msk.nii.gz', 'sub-strokecase0064_ses-0001_msk.nii.gz', 'sub-strokecase0067_ses-0001_msk.nii.gz', 'sub-strokecase0074_ses-0001_msk.nii.gz', 'sub-strokecase0076_ses-0001_msk.nii.gz', 'sub-strokecase0080_ses-0001_msk.nii.gz', 'sub-strokecase0082_ses-0001_msk.nii.gz', 'sub-strokecase0084_ses-0001_msk.nii.gz', 'sub-strokecase0090_ses-0001_msk.nii.gz', 'sub-strokecase0095_ses-0001_msk.nii.gz', 'sub-strokecase0097_ses-0001_msk.nii.gz', 'sub-strokecase0108_ses-0001_msk.nii.gz', 'sub-strokecase0110_ses-0001_msk.nii.gz', 'sub-strokecase0129_ses-0001_msk.nii.gz', 'sub-strokecase0137_ses-0001_msk.nii.gz', 'sub-strokecase0145_ses-0001_msk.nii.gz', 'sub-strokecase0152_ses-0001_msk.nii.gz', 'sub-strokecase0158_ses-0001_msk.nii.gz', 'sub-strokecase0159_ses-0001_msk.nii.gz', 'sub-strokecase0163_ses-0001_msk.nii.gz', 'sub-strokecase0167_ses-0001_msk.nii.gz', 'sub-strokecase0169_ses-0001_msk.nii.gz', 'sub-strokecase0182_ses-0001_msk.nii.gz', 'sub-strokecase0183_ses-0001_msk.nii.gz', 'sub-strokecase0185_ses-0001_msk.nii.gz', 'sub-strokecase0187_ses-0001_msk.nii.gz', 'sub-strokecase0193_ses-0001_msk.nii.gz', 'sub-strokecase0196_ses-0001_msk.nii.gz', 'sub-strokecase0197_ses-0001_msk.nii.gz', 'sub-strokecase0200_ses-0001_msk.nii.gz', 'sub-strokecase0210_ses-0001_msk.nii.gz', 'sub-strokecase0214_ses-0001_msk.nii.gz', 'sub-strokecase0218_ses-0001_msk.nii.gz', 'sub-strokecase0004_ses-0001_msk.nii.gz', 'sub-strokecase0009_ses-0001_msk.nii.gz', 'sub-strokecase0010_ses-0001_msk.nii.gz', 'sub-strokecase0017_ses-0001_msk.nii.gz', 'sub-strokecase0022_ses-0001_msk.nii.gz', 'sub-strokecase0024_ses-0001_msk.nii.gz', 'sub-strokecase0026_ses-0001_msk.nii.gz', 'sub-strokecase0036_ses-0001_msk.nii.gz', 'sub-strokecase0038_ses-0001_msk.nii.gz', 'sub-strokecase0040_ses-0001_msk.nii.gz', 'sub-strokecase0041_ses-0001_msk.nii.gz', 'sub-strokecase0049_ses-0001_msk.nii.gz', 'sub-strokecase0053_ses-0001_msk.nii.gz', 'sub-strokecase0054_ses-0001_msk.nii.gz', 'sub-strokecase0056_ses-0001_msk.nii.gz', 'sub-strokecase0064_ses-0001_msk.nii.gz', 'sub-strokecase0067_ses-0001_msk.nii.gz', 'sub-strokecase0074_ses-0001_msk.nii.gz', 'sub-strokecase0076_ses-0001_msk.nii.gz', 'sub-strokecase0080_ses-0001_msk.nii.gz', 'sub-strokecase0082_ses-0001_msk.nii.gz', 'sub-strokecase0084_ses-0001_msk.nii.gz', 'sub-strokecase0090_ses-0001_msk.nii.gz', 'sub-strokecase0095_ses-0001_msk.nii.gz', 'sub-strokecase0097_ses-0001_msk.nii.gz', 'sub-strokecase0108_ses-0001_msk.nii.gz', 'sub-strokecase0110_ses-0001_msk.nii.gz', 'sub-strokecase0129_ses-0001_msk.nii.gz', 'sub-strokecase0137_ses-0001_msk.nii.gz', 'sub-strokecase0145_ses-0001_msk.nii.gz', 'sub-strokecase0152_ses-0001_msk.nii.gz', 'sub-strokecase0158_ses-0001_msk.nii.gz', 'sub-strokecase0159_ses-0001_msk.nii.gz', 'sub-strokecase0163_ses-0001_msk.nii.gz', 'sub-strokecase0167_ses-0001_msk.nii.gz', 'sub-strokecase0169_ses-0001_msk.nii.gz', 'sub-strokecase0182_ses-0001_msk.nii.gz', 'sub-strokecase0183_ses-0001_msk.nii.gz', 'sub-strokecase0185_ses-0001_msk.nii.gz', 'sub-strokecase0187_ses-0001_msk.nii.gz', 'sub-strokecase0193_ses-0001_msk.nii.gz', 'sub-strokecase0196_ses-0001_msk.nii.gz', 'sub-strokecase0197_ses-0001_msk.nii.gz', 'sub-strokecase0200_ses-0001_msk.nii.gz', 'sub-strokecase0210_ses-0001_msk.nii.gz', 'sub-strokecase0214_ses-0001_msk.nii.gz', 'sub-strokecase0218_ses-0001_msk.nii.gz', 'sub-strokecase0225_ses-0001_msk.nii.gz', 'sub-strokecase0229_ses-0001_msk.nii.gz', 'sub-strokecase0232_ses-0001_msk.nii.gz', 'sub-strokecase0235_ses-0001_msk.nii.gz', 'sub-strokecase0244_ses-0001_msk.nii.gz', 'sub-strokecase0247_ses-0001_msk.nii.gz', 'sub-strokecase0249_ses-0001_msk.nii.gz']
        self.small_group_adc_images = [self.root_dir+"datasets/final_adc_dataset_small/ISLES_registered/"+filename.replace("msk", "adc") for filename in small_group]
        self.small_group_flair_images = [self.root_dir+"datasets/final_flair_dataset_small/isles_registered/"+filename.replace("msk", "FLAIR") for filename in small_group]
        self.small_group_masks = [self.root_dir+"datasets/final_adc_dataset_small/ISLES_masks_registered/"+filename for filename in small_group]

        test_anomaly_transforms = define_instance(args, "val_transforms")

        if "flair" in args.dataset["name"].lower():
            
            self.large_group_masks = [path for path in self.large_group_masks if "0222_ses-0001" not in path]
            
            self.test_anomaly_large_ds = CacheDataset(data=self.large_group_flair_images, transform=test_anomaly_transforms)
            self.test_anomaly_medium_ds = CacheDataset(data=self.medium_group_flair_images, transform=test_anomaly_transforms)
            self.test_anomaly_small_ds = CacheDataset(data=self.small_group_flair_images, transform=test_anomaly_transforms)
        
        elif "adc" in args.dataset["name"].lower():
            
            self.test_anomaly_large_ds = CacheDataset(data=self.large_group_adc_images, transform=test_anomaly_transforms)
            self.test_anomaly_medium_ds = CacheDataset(data=self.medium_group_adc_images, transform=test_anomaly_transforms)
            self.test_anomaly_small_ds = CacheDataset(data=self.small_group_adc_images, transform=test_anomaly_transforms)


        # dataloaders
        # each group of images is split into two halves:
        # select params half: used to select the best noise timestep value, best threshold etc
        # metrics half: used to compute the final scores (e.g DICE) with these best values.

        # images
        test_anomaly_large_ds = CacheDataset(data=self.test_anomaly_large_images, transform=test_anomaly_transforms)
        test_anomaly_medium_ds = CacheDataset(data=self.test_anomaly_medium_images, transform=test_anomaly_transforms)
        test_anomaly_small_ds = CacheDataset(data=self.test_anomaly_small_images, transform=test_anomaly_transforms)
        
        # large group
        self.test_anomaly_large_loader_select_params = DataLoader( 
            _subset_first_half(test_anomaly_large_ds), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        self.test_anomaly_large_images_select_params = self.test_anomaly_large_images[:len(self.test_anomaly_large_images)//2]

        self.test_anomaly_large_loader_metrics = DataLoader(       
            _subset_second_half(test_anomaly_large_ds), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        self.test_anomaly_large_images_metrics = self.test_anomaly_large_images[len(self.test_anomaly_large_images)//2:]

        # medium group
        self.test_anomaly_medium_loader_select_params = DataLoader(
            _subset_first_half(test_anomaly_medium_ds), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        self.test_anomaly_medium_images_select_params = self.test_anomaly_medium_images[:len(self.test_anomaly_medium_images)//2]

        self.test_anomaly_medium_loader_metrics = DataLoader(
            _subset_second_half(test_anomaly_medium_ds), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        self.test_anomaly_medium_images_metrics = self.test_anomaly_medium_images[len(self.test_anomaly_medium_images)//2:]

        # small group
        self.test_anomaly_small_loader_select_params = DataLoader(
            _subset_first_half(test_anomaly_small_ds), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        self.test_anomaly_small_images_select_params = self.test_anomaly_small_images[:len(self.test_anomaly_small_images)//2]

        self.test_anomaly_small_loader_metrics = DataLoader(
            _subset_second_half(test_anomaly_small_ds), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        self.test_anomaly_small_images_metrics = self.test_anomaly_small_images[len(self.test_anomaly_small_images)//2:]
        
        # masks
        self.test_masks_large_ds = CacheDataset(data=self.large_group_masks, transform=masks_transforms)
        self.test_masks_medium_ds = CacheDataset(data=self.medium_group_masks, transform=masks_transforms)
        self.test_masks_small_ds = CacheDataset(data=self.small_group_masks, transform=masks_transforms)

        # large group
        self.test_masks_large_loader_select_params = DataLoader(
            _subset_first_half(self.test_masks_large_ds), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        self.test_masks_large_loader_metrics = DataLoader(
            _subset_second_half(self.test_masks_large_ds), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )

        # medium group
        self.test_masks_medium_loader_select_params = DataLoader(
            _subset_first_half(self.test_masks_medium_ds), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        self.test_masks_medium_loader_metrics = DataLoader(
            _subset_second_half(self.test_masks_medium_ds), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )

        # small group
        self.test_masks_small_loader_select_params = DataLoader(
            _subset_first_half(self.test_masks_small_ds), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        self.test_masks_small_loader_metrics = DataLoader(
            _subset_second_half(self.test_masks_small_ds), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )

    def len_large_group(self):
        return len(self.test_anomaly_large_images)
    
    def len_medium_group(self):
        return len(self.test_anomaly_medium_images)
    
    def len_small_group(self):
        return len(self.test_anomaly_small_images)
    
    def first(self):
        return first(self.test_anomaly_large_loader_select_params)
    
        
    def get_anomaly_loader_select_params(self, group):
        if group == "large":
            return self.test_anomaly_large_loader_select_params
        elif group == "medium":
            return self.test_anomaly_medium_loader_select_params
        elif group == "small":
            return self.test_anomaly_small_loader_select_params
        else:
            raise ValueError("Invalid group name. Must be 'large', 'medium', or 'small'.")
    
    def get_anomaly_loader_metrics(self, group):
        if group == "large":
            return self.test_anomaly_large_loader_metrics
        elif group == "medium":
            return self.test_anomaly_medium_loader_metrics
        elif group == "small":
            return self.test_anomaly_small_loader_metrics
        else:
            raise ValueError("Invalid group name. Must be 'large', 'medium', or 'small'.")
    
    def get_masks_loader_select_params(self, group):
        if group == "large":
            return self.test_masks_large_loader_select_params
        elif group == "medium":
            return self.test_masks_medium_loader_select_params
        elif group == "small":
            return self.test_masks_small_loader_select_params
        else:
            raise ValueError("Invalid group name. Must be 'large', 'medium', or 'small'.")
    
    def get_masks_loader_metrics(self, group):
        if group == "large":
            return self.test_masks_large_loader_metrics
        elif group == "medium":
            return self.test_masks_medium_loader_metrics
        elif group == "small":
            return self.test_masks_small_loader_metrics
        else:
            raise ValueError("Invalid group name. Must be 'large', 'medium', or 'small'.")

    def get_anomaly_images_select_params(self, group):
        if group == "large":
            return self.test_anomaly_large_images_select_params
        elif group == "medium":
            return self.test_anomaly_medium_images_select_params
        elif group == "small":
            return self.test_anomaly_small_images_select_params
        else:
            raise ValueError("Invalid group name. Must be 'large', 'medium', or 'small'.")
    
    def get_anomaly_images_metrics(self, group):
        if group == "large":
            return self.test_anomaly_large_images_metrics
        elif group == "medium":
            return self.test_anomaly_medium_images_metrics
        elif group == "small":
            return self.test_anomaly_small_images_metrics
        else:
            raise ValueError("Invalid group name. Must be 'large', 'medium', or 'small'.")



class BRATS():
    # TODO finish this class
    def __init__(self, 
                 args,
                 batch_size=64,
                 num_workers=4,
                ):
        
        self.args = args
        self.root_dir = args.root_dir
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers

        masks_transforms = transforms.Compose(
            [
                transforms.LoadImage(),
                transforms.EnsureChannelFirst(),
                transforms.ResizeWithPadOrCrop(spatial_size=(args.image_size, args.image_size, args.image_size)),
                custom_transforms.SetBackgroundToZero()
            ]
         )
    
        test_anomaly_images = sorted(glob.glob(self.root_dir+"datasets/final_flair_dataset_small/brats_registered/*.nii.gz"))[:300] #otherwise there are too many images (1200)
        test_masks = sorted(glob.glob(self.root_dir+"datasets/final_flair_dataset_small/brats_masks_registered/*.nii.gz"))[:300] # TODO

        # Read the CSV file and put every line in a list
        masks_to_exclude = []
        
        with open(self.root_dir+"AnoDiffExperiments/data_splits_lists/final_flair_dataset_small/exclude_brats_middle_slice.csv", 'r') as f:
            for line in f:
                masks_to_exclude.append(line.strip())
        images_to_exclude = [name.replace("seg", "t2f") for name in masks_to_exclude]

        test_anomaly_images = [path for path in test_anomaly_images if os.path.basename(path) not in images_to_exclude]
        test_masks = [path for path in test_masks if os.path.basename(path) not in masks_to_exclude]
        #print(test_anomaly_images)

        num_workers = 4
        ano_batch_size = 32

        test_anomaly_transforms = define_instance(args, "val_transforms")
        test_anomaly_ds = CacheDataset(data=test_anomaly_images, transform=test_anomaly_transforms)

        test_anomaly_loader_select_params = DataLoader( # the first 50% of the test data is used to select the best noise timestep value and best threshold.
            _subset_first_half(test_anomaly_ds), batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_anomaly_images_select_params = test_anomaly_images[:len(test_anomaly_ds)//2]

        test_anomaly_loader_metrics = DataLoader(       # The second 50% is used to compute the final IOU and DICE metrics with these best values.
            _subset_second_half(test_anomaly_ds), batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_anomaly_images_metrics = test_anomaly_images[len(test_anomaly_ds)//2:]

        test_masks_ds = CacheDataset(data=test_masks, transform=masks_transforms)
        
        test_masks_loader_select_params = DataLoader( # the first 50% of the test data is used to select the best noise timestep value and best threshold.
            _subset_first_half(test_masks_ds), batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_masks_loader_metrics = DataLoader(       # The second 50% is used to compute the final IOU and DICE metrics with these best values.
            _subset_second_half(test_masks_ds), batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
    
        def first(self):
            return first(self.test_anomaly_loader_select_params)